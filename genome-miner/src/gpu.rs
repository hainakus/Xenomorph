use std::{
    iter::once,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use clap::ArgMatches;
use kaspa_addresses::Address;
use kaspa_consensus_core::header::Header;
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_pow::{
    genome_pow::{
        fragment_index, genome_mix_hash, GenomeDatasetLoader,
        GenomePowState, SyntheticLoader, GENOME_BASE_SIZE, MIX_CHUNK_BYTES,
    },
    matrix::Matrix,
    State as KHeavyState,
};
use kaspa_rpc_core::{
    api::rpc::RpcApi, model::message::GetBlockTemplateRequest, RpcRawBlock,
    SubmitBlockReport, SubmitBlockRejectReason,
};
use tokio::time::sleep;
use wgpu::util::DeviceExt;

// ── GPU pipeline ──────────────────────────────────────────────────────────────

struct GpuContext {
    device:      wgpu::Device,
    queue:       wgpu::Queue,
    pipeline:    wgpu::ComputePipeline,   // Genome PoW
    kh_pipeline: wgpu::ComputePipeline,   // KHeavyHash (pre-activation)
    bind_layout: wgpu::BindGroupLayout,   // shared: 3×storage bindings
}

// ── Per-GPU worker with pre-allocated persistent buffers ─────────────────────
//
// Hot path per dispatch: write_buffer (params) + clear_buffer (output) +
// command encode + submit + poll — zero GPU heap allocations.

struct GpuWorker {
    id:   usize,
    ctx:  Arc<GpuContext>,

    // 739 MB packed genome in VRAM (uploaded once at startup, referenced by g_bind_group)
    #[allow(dead_code)]
    genome_buf: Arc<wgpu::Buffer>,

    // KH: 64×64 matrix (~16 KB). Written once per new pre_pow_hash.
    matrix_buf:        Arc<wgpu::Buffer>,
    last_matrix_hash:  Option<kaspa_hashes::Hash>,

    // Genome PoW persistent resources
    g_params_buf:   wgpu::Buffer,   // 112 bytes
    g_output_buf:   wgpu::Buffer,   // 16 bytes  (STORAGE | COPY_SRC | COPY_DST)
    g_readback_buf: wgpu::Buffer,   // 16 bytes  (MAP_READ | COPY_DST)
    g_bind_group:   wgpu::BindGroup,

    // KHeavyHash persistent resources
    kh_params_buf:   wgpu::Buffer,  // 88 bytes
    kh_output_buf:   wgpu::Buffer,  // 48 bytes  (STORAGE | COPY_SRC | COPY_DST)
    kh_readback_buf: wgpu::Buffer,  // 48 bytes  (MAP_READ | COPY_DST)
    kh_bind_group:   wgpu::BindGroup,
}

impl GpuWorker {
    fn new(id: usize, ctx: Arc<GpuContext>, genome_buf: Arc<wgpu::Buffer>) -> Self {
        let dev = &ctx.device;

        // ── Genome PoW buffers ──
        let g_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g_params"),
            size:  112,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g_output_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g_output"),
            size:  16,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g_readback_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g_readback"),
            size:  16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g_bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("g_bg"),
            layout:  &ctx.bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: g_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: genome_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: g_output_buf.as_entire_binding() },
            ],
        });

        // ── KHeavyHash buffers ──
        let matrix_buf = Arc::new(dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kh_matrix"),
            size:  4096 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let kh_params_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kh_params"),
            size:  88,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kh_output_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kh_output"),
            size:  48,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kh_readback_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kh_readback"),
            size:  48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kh_bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("kh_bg"),
            layout:  &ctx.bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: kh_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: matrix_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kh_output_buf.as_entire_binding() },
            ],
        });

        Self {
            id,
            ctx,
            genome_buf,
            matrix_buf,
            last_matrix_hash: None,
            g_params_buf, g_output_buf, g_readback_buf, g_bind_group,
            kh_params_buf, kh_output_buf, kh_readback_buf, kh_bind_group,
        }
    }
}

// ── Mining template (shared via watch channel) ───────────────────────────────

#[derive(Clone)]
struct MiningTemplate {
    id:             kaspa_hashes::Hash,
    rpc_block:      Arc<RpcRawBlock>,
    header:         Header,
    pre_pow_hash:   kaspa_hashes::Hash,
    target:         kaspa_math::Uint256,
    genome_active:  bool,
    num_mix_chunks: u32,
}

struct Solution {
    nonce:    u64,
    template: Arc<MiningTemplate>,
    gpu_id:   usize,
}

// ── Adapter enumeration ──────────────────────────────────────────────────────

/// On headless Linux (HiveOS, mining rigs) the NVIDIA Vulkan ICD is often installed
/// but not in the Vulkan loader's default search path, so only `llvmpipe` is visible.
/// This function probes common ICD locations and sets `VK_ICD_FILENAMES` so that wgpu
/// finds the real NVIDIA GPU. It is a no-op if the variable is already set by the user.
fn maybe_set_nvidia_vulkan_icd() {
    // Already overridden by the user — respect it.
    if std::env::var_os("VK_ICD_FILENAMES").is_some()
        || std::env::var_os("VK_DRIVER_FILES").is_some()
    {
        return;
    }

    let candidates = [
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "/etc/vulkan/icd.d/nvidia_icd.json",
        "/usr/lib/x86_64-linux-gnu/nvidia/icd/nvidia_icd.json",
        "/usr/lib64/vulkan/icd/nvidia_icd.json",
        "/usr/lib/vulkan/icd/nvidia_icd.json",
    ];

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            // SAFETY: single-threaded init before wgpu Instance is created.
            std::env::set_var("VK_ICD_FILENAMES", path);
            info!("NVIDIA Vulkan ICD auto-detected: {path} (set VK_ICD_FILENAMES to override)");
            return;
        }
    }
}

/// Returns all eligible mining adapters sorted by preference (discrete > integrated).
/// Excludes: software renderers (llvmpipe/lavapipe/softpipe) and Intel integrated GPUs
/// (UHD 600/700) whose max_storage_buffer_binding_size is too small for the 739 MB genome.
pub async fn enumerate_mining_adapters() -> Vec<wgpu::Adapter> {
    maybe_set_nvidia_vulkan_icd();
    const INTEL_VENDOR_ID: u32 = 0x8086;
    let all_backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: all_backends,
        ..Default::default()
    });
    let mut candidates: Vec<wgpu::Adapter> = instance
        .enumerate_adapters(all_backends)
        .into_iter()
        .filter(|a| {
            let info = a.get_info();
            let name_lc = info.name.to_lowercase();
            if info.device_type == wgpu::DeviceType::Cpu
                || name_lc.contains("llvmpipe")
                || name_lc.contains("lavapipe")
                || name_lc.contains("softpipe")
            {
                return false;
            }
            if info.vendor == INTEL_VENDOR_ID
                && info.device_type == wgpu::DeviceType::IntegratedGpu
            {
                warn!("Skipping Intel integrated GPU '{}' (binding size too small for 739 MB genome)", info.name);
                return false;
            }
            true
        })
        .collect();
    candidates.sort_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu   => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        _                               => 2,
    });
    candidates
}

/// Select adapters by `--gpu` value: "all" returns all, otherwise parses comma-separated indices.
pub fn select_adapters(gpu_arg: &str, adapters: Vec<wgpu::Adapter>) -> Vec<wgpu::Adapter> {
    if gpu_arg.trim().eq_ignore_ascii_case("all") {
        return adapters;
    }
    let idx_set: std::collections::HashSet<usize> = gpu_arg
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    adapters.into_iter().enumerate()
        .filter(|(i, _)| idx_set.contains(i))
        .map(|(_, a)| a)
        .collect()
}

impl GpuContext {
    /// Create a `GpuContext` from a specific pre-selected adapter.
    pub async fn from_adapter(adapter: wgpu::Adapter) -> Self {
        info!("GPU: {}", adapter.get_info().name);

        // Request the adapter's actual max buffer size.
        // The WebGPU default (256 MB) is too small for the 739 MB packed genome.
        let adapter_limits = adapter.limits();
        info!("GPU max_buffer_size: {} MB", adapter_limits.max_buffer_size / 1_048_576);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_buffer_size: adapter_limits.max_buffer_size,
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Failed to get GPU device");

        let shader_src = include_str!("genome_pow.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("genome_pow"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let kh_shader_src = include_str!("kheavyhash4.wgsl");
        let kh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("kheavyhash"),
            source: wgpu::ShaderSource::Wgsl(kh_shader_src.into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("genome_pow_bgl"),
            entries: &[
                // binding 0: Params (read-only storage — avoids 16-byte uniform alignment)
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 1: fragment hashes (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // binding 2: output (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty:         wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("genome_pow_pl"),
            bind_group_layouts:   &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("genome_pow_cp"),
            layout:      Some(&pipeline_layout),
            module:      &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let kh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("kheavyhash_cp"),
            layout:      Some(&pipeline_layout),
            module:      &kh_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        Self { device, queue, pipeline, kh_pipeline, bind_layout }
    }

    #[allow(dead_code)]
    async fn new() -> Self {
        let adapters = enumerate_mining_adapters().await;
        let adapter = adapters.into_iter().next()
            .expect("No real GPU adapter found. genome-miner requires Metal, Vulkan, or DX12 (NVIDIA / AMD / Intel Arc / Apple).");
        Self::from_adapter(adapter).await
    }
}

// ── Packed genome helpers ────────────────────────────────────────────────────

/// Build a synthetic packed genome for devnet/testing (no real file).
fn synthetic_packed_genome(frag_size: u32) -> Vec<u8> {
    let num_fragments = (GENOME_BASE_SIZE / frag_size.max(1) as u64) as usize;
    let packed_frag = frag_size as usize / 4;
    (0..num_fragments).flat_map(|i| std::iter::repeat((i & 0xFF) as u8).take(packed_frag)).collect()
}

// ── Param buffer builders ────────────────────────────────────────────────────

/// KHeavyHash matrix → 4096×u32 LE bytes (~16 KB).
fn build_matrix_bytes(pre_pow_hash: &kaspa_hashes::Hash) -> Vec<u8> {
    let matrix = Matrix::generate(*pre_pow_hash);
    let flat = matrix.to_flat_u32();
    flat.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 88-byte KHeavyHash params buffer.
fn build_kheavy_params(
    pre_pow_hash: &kaspa_hashes::Hash,
    timestamp: u64,
    target: &kaspa_math::Uint256,
    nonce_base: u64,
) -> [u8; 88] {
    let mut buf = [0u8; 88];
    buf[0..32].copy_from_slice(pre_pow_hash.as_ref());
    buf[32..36].copy_from_slice(&(timestamp as u32).to_le_bytes());
    buf[36..40].copy_from_slice(&((timestamp >> 32) as u32).to_le_bytes());
    buf[40..72].copy_from_slice(&target.to_le_bytes());
    buf[72..76].copy_from_slice(&(nonce_base as u32).to_le_bytes());
    buf[76..80].copy_from_slice(&((nonce_base >> 32) as u32).to_le_bytes());
    buf
}

/// 112-byte Genome PoW params buffer matching the WGSL Params struct.
fn build_params_full(
    epoch_seed: &kaspa_hashes::Hash,
    pre_pow_hash: &kaspa_hashes::Hash,
    target: &kaspa_math::Uint256,
    nonce_base: u64,
    num_mix_chunks: u32,
) -> [u8; 112] {
    let mut buf = [0u8; 112];
    buf[0..32].copy_from_slice(epoch_seed.as_ref());
    buf[32..64].copy_from_slice(pre_pow_hash.as_ref());
    buf[64..96].copy_from_slice(&target.to_le_bytes());
    buf[96..100].copy_from_slice(&(nonce_base as u32).to_le_bytes());
    buf[100..104].copy_from_slice(&((nonce_base >> 32) as u32).to_le_bytes());
    buf[104..108].copy_from_slice(&num_mix_chunks.to_le_bytes());
    buf
}

// ── Per-batch GPU dispatch (zero-alloc hot path) ─────────────────────────────

/// Genome PoW dispatch using pre-allocated persistent buffers.
/// Hot path: write_buffer (params) + clear_buffer (output) + encode + submit + poll.
async fn dispatch_genome(worker: &mut GpuWorker, params_data: &[u8; 112], batch_size: u32) -> Option<u64> {
    let dev   = &worker.ctx.device;
    let queue = &worker.ctx.queue;

    queue.write_buffer(&worker.g_params_buf, 0, params_data);

    let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&worker.g_output_buf, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&worker.ctx.pipeline);
        pass.set_bind_group(0, &worker.g_bind_group, &[]);
        pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&worker.g_output_buf, 0, &worker.g_readback_buf, 0, 16);
    queue.submit(once(encoder.finish()));

    let slice = worker.g_readback_buf.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    dev.poll(wgpu::Maintain::Wait);
    rx.await.ok()?.ok()?;

    let data     = slice.get_mapped_range();
    let found    = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let nonce_lo = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let nonce_hi = u32::from_le_bytes(data[8..12].try_into().unwrap());
    drop(data);
    worker.g_readback_buf.unmap();

    if found != 0 { Some((nonce_lo as u64) | ((nonce_hi as u64) << 32)) } else { None }
}

/// KHeavyHash dispatch using pre-allocated persistent buffers.
/// Returns `Some((nonce, gpu_hash))` so the caller can CPU-verify.
async fn dispatch_kheavy(worker: &mut GpuWorker, params_data: &[u8; 88], batch_size: u32) -> Option<(u64, [u32; 8])> {
    let dev   = &worker.ctx.device;
    let queue = &worker.ctx.queue;

    queue.write_buffer(&worker.kh_params_buf, 0, params_data);

    let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&worker.kh_output_buf, 0, None);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&worker.ctx.kh_pipeline);
        pass.set_bind_group(0, &worker.kh_bind_group, &[]);
        pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&worker.kh_output_buf, 0, &worker.kh_readback_buf, 0, 48);
    queue.submit(once(encoder.finish()));

    let slice = worker.kh_readback_buf.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    dev.poll(wgpu::Maintain::Wait);
    rx.await.ok()?.ok()?;

    let data     = slice.get_mapped_range();
    let found    = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let nonce_lo = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let nonce_hi = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let mut gpu_hash = [0u32; 8];
    for i in 0..8 {
        gpu_hash[i] = u32::from_le_bytes(data[16 + i * 4..16 + i * 4 + 4].try_into().unwrap());
    }
    drop(data);
    worker.kh_readback_buf.unmap();

    if found != 0 { Some(((nonce_lo as u64) | ((nonce_hi as u64) << 32), gpu_hash)) } else { None }
}

// ── Per-GPU mining task (runs as an independent tokio::spawn) ─────────────────
//
// Each GPU owns its worker and independently dispatches batches, reading the
// latest template from a watch channel. Template changes cause nonce reset.
// Multiple GPUs mine in parallel without blocking each other.

async fn gpu_mining_task(
    mut worker:      GpuWorker,
    template_rx:     tokio::sync::watch::Receiver<Option<Arc<MiningTemplate>>>,
    solution_tx:     tokio::sync::mpsc::Sender<Solution>,
    hash_counter:    Arc<AtomicU64>,
    batch_size:      u32,
    num_gpus:        usize,
) {
    let gpu_id = worker.id;
    let mut current_id: Option<kaspa_hashes::Hash> = None;
    // Interleave starting nonces: GPU i starts at i * batch_size
    let mut nonce_base: u64 = gpu_id as u64 * batch_size as u64;

    loop {
        let template: Option<Arc<MiningTemplate>> = template_rx.borrow().clone();
        let template = match template {
            Some(t) => t,
            None    => { sleep(Duration::from_millis(20)).await; continue; }
        };

        if Some(template.id) != current_id {
            current_id    = Some(template.id);
            nonce_base    = gpu_id as u64 * batch_size as u64;
        }

        let result: Option<u64> = if template.genome_active {
            let params = build_params_full(
                &template.header.epoch_seed, &template.pre_pow_hash,
                &template.target, nonce_base, template.num_mix_chunks,
            );
            dispatch_genome(&mut worker, &params, batch_size).await
        } else {
            if worker.last_matrix_hash != Some(template.pre_pow_hash) {
                worker.ctx.queue.write_buffer(&worker.matrix_buf, 0, &build_matrix_bytes(&template.pre_pow_hash));
                worker.last_matrix_hash = Some(template.pre_pow_hash);
            }
            let kh_params = build_kheavy_params(
                &template.pre_pow_hash, template.header.timestamp, &template.target, nonce_base,
            );
            match dispatch_kheavy(&mut worker, &kh_params, batch_size).await {
                Some((nonce, gpu_hash)) => {
                    let state = KHeavyState::new(&template.header);
                    let (cpu_valid, cpu_pow) = state.check_pow(nonce);
                    let cpu_bytes = cpu_pow.to_le_bytes();
                    let mut cpu_hash = [0u32; 8];
                    for k in 0..8 { cpu_hash[k] = u32::from_le_bytes(cpu_bytes[k*4..k*4+4].try_into().unwrap()); }
                    if gpu_hash != cpu_hash {
                        warn!("[GPU{}] KHeavyHash mismatch nonce={:#018x}", gpu_id, nonce);
                    }
                    if cpu_valid { Some(nonce) } else {
                        warn!("[GPU{}] KHeavyHash false-positive nonce={:#018x} — skipping", gpu_id, nonce);
                        None
                    }
                }
                None => None,
            }
        };

        hash_counter.fetch_add(batch_size as u64, Ordering::Relaxed);
        nonce_base = nonce_base.wrapping_add(num_gpus as u64 * batch_size as u64);

        if let Some(nonce) = result {
            if solution_tx.send(Solution { nonce, template, gpu_id }).await.is_err() {
                return; // main task exited
            }
        }
    }
}

// ── `gpu` subcommand entry point ──────────────────────────────────────────────

pub async fn cmd_gpu(m: &ArgMatches) {
    let rpcserver         = m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:36669".to_owned());
    let addr_str          = m.get_one::<String>("mining-address").cloned().expect("--mining-address required");
    let batch_size        = m.get_one::<u32>("batch-size").copied().unwrap_or(1 << 20);
    let frag_size         = m.get_one::<u32>("genome-fragment-size").copied().unwrap_or(1_048_576);
    let genome_activation = crate::resolve_activation(m);
    let gpu_arg           = m.get_one::<String>("gpu").cloned().unwrap_or_else(|| "0".to_owned());
    let list_gpus         = m.get_flag("list-gpus");
    let genome_path: Option<String> = m.get_one::<String>("genome-file").cloned().or_else(|| {
        let default = dirs::home_dir()?.join(".rusty-xenom").join("grch38.xenom");
        if default.exists() { Some(default.to_string_lossy().into_owned()) } else { None }
    });

    // Enumerate eligible adapters
    let all_adapters = enumerate_mining_adapters().await;

    if list_gpus {
        if all_adapters.is_empty() {
            info!("No eligible GPU adapters found (software renderers and Intel iGPU excluded).");
        } else {
            info!("{} eligible GPU adapter(s):", all_adapters.len());
            for (i, a) in all_adapters.iter().enumerate() {
                let inf = a.get_info();
                info!("  [{}] {} — {:?} (vendor: {:#06x})", i, inf.name, inf.device_type, inf.vendor);
            }
        }
        return;
    }

    let selected = select_adapters(&gpu_arg, all_adapters);
    if selected.is_empty() {
        warn!("No GPUs matched '--gpu {gpu_arg}'. Run with --list-gpus to see available indices.");
        return;
    }

    // Load genome dataset once; bytes uploaded to each GPU's VRAM separately.
    let file_loader: Option<kaspa_pow::genome_file::FileGenomeLoader> =
        genome_path.as_deref().map(|path| {
            kaspa_pow::genome_file::FileGenomeLoader::open(std::path::Path::new(path), frag_size, false)
                .unwrap_or_else(|e| panic!("Failed to open genome file '{path}': {e}"))
        });
    let synthetic_bytes: Vec<u8> =
        if file_loader.is_none() { synthetic_packed_genome(frag_size) } else { Vec::new() };
    // Own the packed bytes so `file_loader` is not borrowed when we later move it into Arc.
    let (packed_genome_bytes, num_mix_chunks): (Vec<u8>, u32) = match file_loader.as_ref() {
        Some(loader) => {
            let packed = loader.packed_dataset().unwrap();
            let chunks = (packed.len() / MIX_CHUNK_BYTES) as u32;
            info!("Genome PoW: loaded {} ({} MB, {chunks} mix-chunks)",
                genome_path.as_deref().unwrap_or(""), packed.len() / 1_048_576);
            (packed.to_vec(), chunks)
        }
        None => {
            let chunks = (synthetic_bytes.len() / MIX_CHUNK_BYTES) as u32;
            info!("Genome PoW: synthetic dataset ({chunks} mix-chunks) — devnet/testing only");
            (synthetic_bytes, chunks)
        }
    };

    // Init GPU workers (pre-allocated persistent buffers + bind groups).
    info!("Initialising {} GPU(s) ...", selected.len());
    let mut workers: Vec<GpuWorker> = Vec::with_capacity(selected.len());
    for (i, adapter) in selected.into_iter().enumerate() {
        let ctx = Arc::new(GpuContext::from_adapter(adapter).await);
        let genome_buf = Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("packed_genome"),
            contents: &packed_genome_bytes,
            usage:    wgpu::BufferUsages::STORAGE,
        }));
        workers.push(GpuWorker::new(i, ctx, genome_buf));
    }
    let num_gpus = workers.len();
    info!("{num_gpus} GPU(s) ready — batch_size={batch_size} nonces/dispatch/GPU");

    let url = format!("grpc://{rpcserver}");
    info!("Connecting to {url}");
    let rpc = Arc::new(GrpcClient::connect(url).await.expect("Failed to connect"));
    let pay_address: kaspa_rpc_core::RpcAddress =
        Address::try_from(addr_str.as_str()).expect("Invalid --mining-address");

    // ── Background template polling ──────────────────────────────────────────
    //
    // A dedicated task polls get_block_template every 200 ms and broadcasts
    // via a watch channel.  GPU tasks read from the watch without blocking RPC.
    let (template_tx, template_rx) =
        tokio::sync::watch::channel::<Option<Arc<MiningTemplate>>>(None);
    {
        let rpc2 = rpc.clone();
        let pay2 = pay_address.clone();
        tokio::spawn(async move {
            loop {
                match rpc2.get_block_template_call(None, GetBlockTemplateRequest::new(pay2.clone(), vec![])).await {
                    Ok(resp) => {
                        if !resp.is_synced {
                            warn!("Node not synced — waiting for IBD to complete");
                            sleep(Duration::from_secs(2)).await;
                            continue;
                        }
                        let block  = resp.block;
                        let id     = block.header.accepted_id_merkle_root;
                        let header: Header = (&block.header).into();
                        let genome_active  = header.daa_score >= genome_activation;
                        let pre_pow_hash = kaspa_consensus_core::hashing::header::hash_override_nonce_time(&header, 0, 0);
                        let target = kaspa_math::Uint256::from_compact_target_bits(header.bits);
                        let t = Arc::new(MiningTemplate {
                            id,
                            rpc_block: Arc::new(block),
                            header,
                            pre_pow_hash,
                            target,
                            genome_active,
                            num_mix_chunks,
                        });
                        // Only wake watchers on actual template change
                        let changed = template_tx.borrow().as_ref().map(|prev| prev.id) != Some(id);
                        if changed {
                            info!("New template daa={}", t.header.daa_score);
                            let _ = template_tx.send(Some(t));
                        }
                    }
                    Err(e) => { warn!("get_block_template: {e}"); sleep(Duration::from_secs(1)).await; }
                }
                sleep(Duration::from_millis(200)).await;
            }
        });
    }

    // ── Per-GPU solution channel ──────────────────────────────────────────────
    let (sol_tx, mut sol_rx) = tokio::sync::mpsc::channel::<Solution>(num_gpus * 2);

    // ── Per-GPU hash counters ─────────────────────────────────────────────────
    let hash_counters: Vec<Arc<AtomicU64>> = (0..num_gpus)
        .map(|_| Arc::new(AtomicU64::new(0)))
        .collect();

    // ── Spawn one mining task per GPU ─────────────────────────────────────────
    //
    // Each task owns its GpuWorker (and therefore its wgpu::Device + buffers).
    // They run concurrently on separate tokio worker threads, dispatching
    // batches independently — no inter-GPU lock contention.
    for worker in workers {
        let rx  = template_rx.clone();
        let tx  = sol_tx.clone();
        let hc  = hash_counters[worker.id].clone();
        tokio::spawn(async move {
            gpu_mining_task(worker, rx, tx, hc, batch_size, num_gpus).await;
        });
    }
    drop(sol_tx); // close when all GPU tasks exit

    // ── Main loop: report hashrate + handle solutions ─────────────────────────
    let file_loader = Arc::new(file_loader);
    let packed_gpu: Arc<Vec<u8>> = Arc::new(packed_genome_bytes);
    let mut report_timer = Instant::now();

    loop {
        // Drain any pending solutions
        loop {
            match sol_rx.try_recv() {
                Ok(sol) => {
                    handle_solution(
                        sol, &rpc, &file_loader, &packed_gpu,
                        frag_size, genome_activation,
                    ).await;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        if report_timer.elapsed() >= Duration::from_secs(5) {
            let total: u64 = hash_counters.iter().map(|c| c.swap(0, Ordering::Relaxed)).sum();
            let mhs = total as f64 / report_timer.elapsed().as_secs_f64() / 1_000_000.0;
            let mode = template_rx.borrow().as_ref().map(|t| if t.genome_active { "Genome" } else { "KHeavyHash" }).unwrap_or("—");
            info!("GPU×{num_gpus} [{mode}] [{mhs:.2} MH/s]");
            report_timer = Instant::now();
        }

        sleep(Duration::from_millis(2)).await;
    }
}

// ── Solution cross-check + submission ────────────────────────────────────────

async fn handle_solution(
    sol:              Solution,
    rpc:              &Arc<GrpcClient>,
    file_loader:      &Arc<Option<kaspa_pow::genome_file::FileGenomeLoader>>,
    packed_bytes:     &Arc<Vec<u8>>,
    frag_size:        u32,
    genome_activation: u64,
) {
    let Solution { nonce, template, gpu_id } = sol;
    let header = &template.header;

    if header.daa_score >= genome_activation {
        let cpu_pow = genome_mix_hash(packed_bytes, &header.epoch_seed, nonce, &template.pre_pow_hash);
        let state   = GenomePowState::new(template.pre_pow_hash, template.target.clone(), header.epoch_seed, frag_size);
        if cpu_pow > state.target {
            warn!("[GPU{}] Genome PoW false-positive nonce={:#018x} — skipping", gpu_id, nonce);
            return;
        }
        let frag_idx = fragment_index(&header.epoch_seed, nonce, frag_size);
        let synth    = SyntheticLoader::new(frag_size, header.epoch_seed);
        let fl: &dyn GenomeDatasetLoader = match file_loader.as_ref() {
            Some(f) => f,
            None    => &synth,
        };
        let fragment  = fl.load_fragment(frag_idx).unwrap_or_else(|| vec![0u8; frag_size as usize]);
        let (_, _, fitness) = state.check_pow_with_fragment(nonce, &fragment);
        info!("[GPU{}] Genome PoW PASS nonce={:#018x} fitness={}", gpu_id, nonce, fitness);
    }

    match rpc.submit_block(build_raw_block_nonce(&template.rpc_block, nonce), false).await {
        Ok(r) => {
            let is_ibd = matches!(r.report, SubmitBlockReport::Reject(SubmitBlockRejectReason::IsInIBD));
            info!("[GPU{}] submit_block: {:?}", gpu_id, r.report);
            if is_ibd { warn!("Node still in IBD"); }
        }
        Err(e) => warn!("[GPU{}] submit_block error: {e}", gpu_id),
    }
}

// Helper: inject winning nonce into a raw block
pub fn build_raw_block_nonce(template: &RpcRawBlock, nonce: u64) -> RpcRawBlock {
    use kaspa_rpc_core::RpcRawHeader;
    let raw_header = RpcRawHeader {
        version:                 template.header.version,
        parents_by_level:        template.header.parents_by_level.clone(),
        hash_merkle_root:        template.header.hash_merkle_root,
        accepted_id_merkle_root: template.header.accepted_id_merkle_root,
        utxo_commitment:         template.header.utxo_commitment,
        timestamp:               template.header.timestamp,
        bits:                    template.header.bits,
        nonce,
        daa_score:               template.header.daa_score,
        blue_work:               template.header.blue_work,
        blue_score:              template.header.blue_score,
        epoch_seed:              template.header.epoch_seed,
        pruning_point:           template.header.pruning_point,
    };
    RpcRawBlock { header: raw_header, transactions: template.transactions.clone() }
}
