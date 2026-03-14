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
use crate::tui::{DashStats, GpuStats};
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
    name: String,
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
    fn new(id: usize, name: String, ctx: Arc<GpuContext>, genome_buf: Arc<wgpu::Buffer>) -> Self {
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
            name,
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

/// Source of a mining template — determines how solutions are submitted.
#[derive(Clone)]
pub(crate) enum MiningSource {
    /// Direct node connection: solution submitted via gRPC submit_block.
    Node { rpc_block: Arc<RpcRawBlock>, header: Header },
    /// Stratum pool: solution submitted as mining.submit extranonce2.
    Stratum { job_id: String, extranonce1: u32 },
}

#[derive(Clone)]
pub(crate) struct MiningTemplate {
    pub id:             kaspa_hashes::Hash,
    pub source:         MiningSource,
    pub pre_pow_hash:   kaspa_hashes::Hash,
    pub epoch_seed:     kaspa_hashes::Hash,
    pub timestamp:      u64,
    pub daa_score:      u64,
    pub bits:           u32,
    pub target:         kaspa_math::Uint256,
    pub genome_active:  bool,
    pub num_mix_chunks: u32,
}

struct Solution {
    nonce:    u64,
    template: Arc<MiningTemplate>,
    gpu_id:   usize,
}

/// Where to send a solved nonce.
pub(crate) enum SolutionSink {
    Node(Arc<GrpcClient>),
    Stratum(tokio::sync::mpsc::Sender<crate::stratum_client::StratumSolution>),
}

// ── Adapter enumeration ──────────────────────────────────────────────────────

/// On headless Linux (HiveOS, mining rigs) the NVIDIA Vulkan ICD is often installed
/// but not in the Vulkan loader's default search path, so only `llvmpipe` is visible.
/// This function probes common ICD locations and sets `VK_ICD_FILENAMES` so that wgpu
/// finds the real NVIDIA GPU. It is a no-op if the variable is already set by the user.
/// On Linux, NVIDIA's Vulkan driver creates ~500-1000 mmap regions per GPU.
/// With many GPUs the default vm.max_map_count (65536) is easily exceeded,
/// causing the kernel to kill the miner with no error message.
#[cfg(target_os = "linux")]
fn check_linux_mmap_limit(n_gpus: usize) {
    if let Ok(val) = std::fs::read_to_string("/proc/sys/vm/max_map_count") {
        let current: u64 = val.trim().parse().unwrap_or(0);
        let needed:  u64 = n_gpus as u64 * 1_000;
        if current < needed.max(262_144) {
            warn!(
                "vm.max_map_count={current} is too low for {n_gpus} GPU(s) (need ~{needed}).\
                \n  NVIDIA Vulkan uses ~500-1000 mmap regions per GPU — process may be killed!\
                \n  Fix (immediate): sudo sysctl -w vm.max_map_count=1048576\
                \n  Fix (persist):   echo 'vm.max_map_count=1048576' | sudo tee -a /etc/sysctl.conf"
            );
        }
    }
}
#[cfg(not(target_os = "linux"))]
fn check_linux_mmap_limit(_n_gpus: usize) {}

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
    pub async fn from_adapter(adapter: wgpu::Adapter) -> Result<Self, String> {
        let gpu_name = adapter.get_info().name;
        info!("GPU: {gpu_name}");

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
            .map_err(|e| format!("[{gpu_name}] request_device failed: {e}"))?;

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

        Ok(Self { device, queue, pipeline, kh_pipeline, bind_layout })
    }

    #[allow(dead_code)]
    async fn new() -> Self {
        let adapters = enumerate_mining_adapters().await;
        let adapter = adapters.into_iter().next()
            .expect("No real GPU adapter found. genome-miner requires Metal, Vulkan, or DX12 (NVIDIA / AMD / Intel Arc / Apple).");
        Self::from_adapter(adapter).await.expect("Failed to init GPU context")
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
        pass.dispatch_workgroups(batch_size.div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&worker.g_output_buf, 0, &worker.g_readback_buf, 0, 16);
    queue.submit(once(encoder.finish()));

    let slice = worker.g_readback_buf.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    tokio::task::block_in_place(|| dev.poll(wgpu::Maintain::Wait));
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
        pass.dispatch_workgroups(batch_size.div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&worker.kh_output_buf, 0, &worker.kh_readback_buf, 0, 48);
    queue.submit(once(encoder.finish()));

    let slice = worker.kh_readback_buf.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    tokio::task::block_in_place(|| dev.poll(wgpu::Maintain::Wait));
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
    nonce_offset:    u64,  // multi-instance offset: instance N passes N here
) {
    let gpu_id = worker.id;
    let mut current_id: Option<kaspa_hashes::Hash> = None;
    // Interleave starting nonces across GPUs; offset by instance index so
    // multiple processes don't mine the same nonce space.
    let stride = num_gpus as u64 * batch_size as u64;
    let base_nonce = |gid: u64| gid * batch_size as u64 + nonce_offset * stride;
    let mut nonce_base: u64 = base_nonce(gpu_id as u64);

    loop {
        let template: Option<Arc<MiningTemplate>> = template_rx.borrow().clone();
        let template = match template {
            Some(t) => t,
            None    => { sleep(Duration::from_millis(20)).await; continue; }
        };

        if Some(template.id) != current_id {
            current_id = Some(template.id);
            nonce_base = match &template.source {
                MiningSource::Stratum { extranonce1, .. } => {
                    (*extranonce1 as u64) << 32 | (gpu_id as u64 * batch_size as u64)
                }
                _ => base_nonce(gpu_id as u64),
            };
        }

        let result: Option<u64> = if template.genome_active {
            let params = build_params_full(
                &template.epoch_seed, &template.pre_pow_hash,
                &template.target, nonce_base, template.num_mix_chunks,
            );
            dispatch_genome(&mut worker, &params, batch_size).await
        } else {
            if worker.last_matrix_hash != Some(template.pre_pow_hash) {
                worker.ctx.queue.write_buffer(&worker.matrix_buf, 0, &build_matrix_bytes(&template.pre_pow_hash));
                worker.last_matrix_hash = Some(template.pre_pow_hash);
            }
            let kh_params = build_kheavy_params(
                &template.pre_pow_hash, template.timestamp, &template.target, nonce_base,
            );
            match dispatch_kheavy(&mut worker, &kh_params, batch_size).await {
                Some((nonce, gpu_hash)) => {
                    match &template.source {
                        MiningSource::Node { header, .. } => {
                            let state = KHeavyState::new(header);
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
                        MiningSource::Stratum { .. } => {
                            // No CPU-verify in stratum mode; bridge validates
                            Some(nonce)
                        }
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

pub async fn cmd_gpu(m: &ArgMatches, dash: std::sync::Arc<std::sync::Mutex<DashStats>>) {
    let rpcserver         = m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:36669".to_owned());
    let addr_str_opt: Option<String> = m.get_one::<String>("mining-address").cloned();
    let batch_size        = m.get_one::<u32>("batch-size").copied().unwrap_or(1 << 20);
    let frag_size         = m.get_one::<u32>("genome-fragment-size").copied().unwrap_or(1_048_576);
    let genome_activation = crate::resolve_activation(m);
    let gpu_arg           = m.get_one::<String>("gpu").cloned().unwrap_or_else(|| "0".to_owned());
    let nonce_offset      = m.get_one::<u64>("nonce-offset").copied().unwrap_or(0);
    let list_gpus         = m.get_flag("list-gpus");
    let genome_path: Option<String> = m.get_one::<String>("genome-file").cloned().or_else(|| {
        let default = dirs::home_dir()?.join(".rusty-xenom").join("grch38.xenom");
        if default.exists() { Some(default.to_string_lossy().into_owned()) } else { None }
    });

    let l2_cfg: Option<crate::l2_worker::L2Config> = match (
        m.get_one::<String>("l2-coordinator").cloned(),
        m.get_one::<String>("l2-private-key").cloned(),
    ) {
        (Some(url), Some(key)) => {
            let use_gpu     = m.get_flag("l2-gpu");
            let perch_script = m.get_one::<String>("l2-perch-script").map(std::path::PathBuf::from);
            match crate::l2_worker::L2Config::new(url, key, use_gpu, perch_script) {
                Ok(c)  => { info!("L2 inline worker enabled — coordinator={}", c.coordinator_url); Some(c) }
                Err(e) => { warn!("L2 config error: {e} — L2 disabled"); None }
            }
        }
        _ => None,
    };

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

    // Check Linux mmap limit before initialising GPUs (NVIDIA Vulkan needs ~1000 mmaps/GPU).
    check_linux_mmap_limit(selected.len());

    // Init GPU workers (pre-allocated persistent buffers + bind groups).
    info!("Initialising {} GPU(s) ...", selected.len());
    let mut workers: Vec<GpuWorker> = Vec::with_capacity(selected.len());
    for (i, adapter) in selected.into_iter().enumerate() {
        let gpu_name = adapter.get_info().name.clone();
        let ctx = match GpuContext::from_adapter(adapter).await {
            Ok(c)  => Arc::new(c),
            Err(e) => {
                warn!("Skipping GPU {i} ({gpu_name}): {e}");
                dash.lock().unwrap().push_log(format!("GPU {i} init failed — skipped: {e}"));
                continue;
            }
        };
        let genome_buf = Arc::new(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("packed_genome"),
            contents: &packed_genome_bytes,
            usage:    wgpu::BufferUsages::STORAGE,
        }));
        workers.push(GpuWorker::new(i, gpu_name, ctx, genome_buf));
    }
    if workers.is_empty() {
        warn!("No GPU workers could be initialised — exiting.");
        return;
    }
    let num_gpus = workers.len();
    {
        let mut s = dash.lock().unwrap();
        s.gpus = workers.iter().map(|w| GpuStats {
            id:       w.id,
            name:     w.name.clone(),
            hashrate: 0.0,
            accepted: 0,
            rejected: 0,
        }).collect();
        s.connected = true;
        s.push_log(format!("{num_gpus} GPU(s) ready — batch={batch_size}"));
    }
    info!("{num_gpus} GPU(s) ready — batch_size={batch_size} nonces/dispatch/GPU");

    let stratum_url: Option<String> = m.get_one::<String>("stratum").cloned();

    // In node mode --mining-address is mandatory; in stratum mode it is optional
    // (the pool handles payouts; --stratum-worker identifies the miner).
    if stratum_url.is_none() && addr_str_opt.is_none() {
        warn!("--mining-address is required in node mode");
        return;
    }
    let addr_str = addr_str_opt.clone().unwrap_or_default();

    let stratum_worker = m.get_one::<String>("stratum-worker")
        .cloned()
        .or_else(|| addr_str_opt.clone())
        .unwrap_or_else(|| "xenom-miner".to_owned());
    let stratum_password = m.get_one::<String>("stratum-password")
        .cloned()
        .unwrap_or_else(|| "x".to_owned());

    // ── Template watch channel (common to both node and stratum modes) ─────
    let (template_tx, template_rx) =
        tokio::sync::watch::channel::<Option<Arc<MiningTemplate>>>(None);

    // ── Solution sink + template source ─────────────────────────────────
    // Build the SolutionSink first; for stratum mode this allocates the channel
    // whose Sender is placed inside SolutionSink::Stratum and whose Receiver is
    // passed into the stratum client's run() loop.
    let sol_sink: SolutionSink;
    let stratum_channels: Option<(
        crate::stratum_client::StratumClient,
        tokio::sync::mpsc::Sender<crate::stratum_client::StratumJob>,
        tokio::sync::mpsc::Receiver<crate::stratum_client::StratumJob>,
        tokio::sync::mpsc::Receiver<crate::stratum_client::StratumSolution>,
    )>;

    if let Some(ref surl) = stratum_url {
        // Stratum mode: solutions go back via a channel to the stratum client
        let (sol_tx_s, sol_rx_s) = tokio::sync::mpsc::channel::<crate::stratum_client::StratumSolution>(32);
        sol_sink = SolutionSink::Stratum(sol_tx_s);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<crate::stratum_client::StratumJob>(8);
        let client = crate::stratum_client::StratumClient::new(surl, &stratum_worker, &stratum_password);
        stratum_channels = Some((client, job_tx, job_rx, sol_rx_s));
    } else {
        let url = format!("grpc://{rpcserver}");
        info!("Connecting to {url}");
        let rpc = Arc::new(GrpcClient::connect(url).await.expect("Failed to connect"));
        sol_sink = SolutionSink::Node(rpc);
        stratum_channels = None;
    };

    // ── Background template source ────────────────────────────────────────
    if let Some((client, job_tx, mut job_rx, sol_rx_s)) = stratum_channels {
        // — Stratum mode: connect to pool and convert jobs to MiningTemplates —
        let dash2 = dash.clone();
        tokio::spawn(async move {
            client.run(job_tx, sol_rx_s, dash2).await;
        });

        // Job-to-template conversion task
        let ttx = template_tx.clone();
        let dash3 = dash.clone();
        let l2_cfg2 = l2_cfg.clone();
        tokio::spawn(async move {
            while let Some(job) = job_rx.recv().await {
                // Dispatch L2 task if coordinator is configured
                if let (Some(ref cfg), Some(ref l2_val)) = (&l2_cfg2, &job.l2_job) {
                    let l2_job_id = l2_val["job_id"].as_str().unwrap_or("").to_owned();
                    if !l2_job_id.is_empty() {
                        let cfg2 = cfg.clone();
                        let val2 = l2_val.clone();
                        tokio::spawn(async move {
                            crate::l2_worker::run_l2_job(cfg2, val2).await;
                        });
                    }
                }
                let target      = kaspa_math::Uint256::from_compact_target_bits(job.bits);
                let genome_active = job.daa_score >= genome_activation;
                let id          = job.pre_pow_hash;
                let extranonce1 = job.extranonce1;
                let job_id      = job.job_id.clone();
                let t = Arc::new(MiningTemplate {
                    id,
                    source: MiningSource::Stratum { job_id, extranonce1 },
                    pre_pow_hash:   job.pre_pow_hash,
                    epoch_seed:     job.epoch_seed,
                    timestamp:      job.timestamp,
                    daa_score:      job.daa_score,
                    bits:           job.bits,
                    target,
                    genome_active,
                    num_mix_chunks,
                });
                let changed = ttx.borrow().as_ref().map(|p| p.id) != Some(id);
                if changed {
                    info!("Stratum job bits={:#010x} genome={}", t.bits, t.genome_active);
                    {
                        let mut s = dash3.lock().unwrap();
                        s.bits          = t.bits;
                        s.genome_active = t.genome_active;
                        s.mode          = format!("GPU×{num_gpus} · Pool Stratum");
                    }
                    ttx.send(Some(t)).ok();
                }
            }
        });
    } else {
        // — Node mode: poll get_block_template via gRPC —
        let rpc = match &sol_sink {
            SolutionSink::Node(r) => r.clone(),
            _ => unreachable!(),
        };
        let pay_address: kaspa_rpc_core::RpcAddress =
            Address::try_from(addr_str.as_str()).expect("Invalid --mining-address");
        let dash2 = dash.clone();
        tokio::spawn(async move {
            loop {
                match rpc.get_block_template_call(None, GetBlockTemplateRequest::new(pay_address.clone(), vec![])).await {
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
                            source: MiningSource::Node {
                                rpc_block: Arc::new(block),
                                header: header.clone(),
                            },
                            pre_pow_hash,
                            epoch_seed: header.epoch_seed,
                            timestamp: header.timestamp,
                            daa_score: header.daa_score,
                            bits: header.bits,
                            target,
                            genome_active,
                            num_mix_chunks,
                        });
                        let changed = template_tx.borrow().as_ref().map(|prev| prev.id) != Some(id);
                        if changed {
                            info!("New template daa={}", t.daa_score);
                            {
                                let mut s = dash2.lock().unwrap();
                                s.daa_score     = t.daa_score;
                                s.bits          = t.bits;
                                s.genome_active = t.genome_active;
                                s.connected     = true;
                                let mode = if t.genome_active { "Genome PoW" } else { "KHeavyHash" };
                                s.mode = format!("GPU×{num_gpus} · {mode}");
                                s.push_log(format!(
                                    "New template daa={} bits={:#010x} genome={}",
                                    t.daa_score, t.bits, t.genome_active
                                ));
                            }
                            template_tx.send(Some(t)).ok();
                        }
                    }
                    Err(e) => { warn!("get_block_template: {e}"); sleep(Duration::from_secs(1)).await; }
                }
                sleep(Duration::from_millis(200)).await;
            }
        });
    }

    // ── Per-GPU solution channel ─────────────────────────────────────────
    let (sol_tx, mut sol_rx) = tokio::sync::mpsc::channel::<Solution>(num_gpus * 2);

    // ── Per-GPU hash counters ────────────────────────────────────────────
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
            gpu_mining_task(worker, rx, tx, hc, batch_size, num_gpus, nonce_offset).await;
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
                        sol, &sol_sink, &file_loader, &packed_gpu,
                        frag_size, &dash,
                    ).await;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        let report_elapsed = report_timer.elapsed();
        if report_elapsed >= Duration::from_secs(5) {
            let elapsed = report_elapsed.as_secs_f64();
            let per_gpu: Vec<f64> = hash_counters.iter()
                .map(|c| c.swap(0, Ordering::Relaxed) as f64 / elapsed / 1_000_000.0)
                .collect();
            let total_mhs: f64 = per_gpu.iter().sum();
            let mode = template_rx.borrow().as_ref()
                .map(|t| if t.genome_active { "Genome" } else { "KHeavyHash" })
                .unwrap_or("—");
            info!("GPU×{num_gpus} [{mode}] [{total_mhs:.2} MH/s]");
            {
                let mut s = dash.lock().unwrap();
                s.total_mhs = total_mhs;
                for (i, &mhs) in per_gpu.iter().enumerate() {
                    if let Some(g) = s.gpus.get_mut(i) { g.hashrate = mhs; }
                }
            }
            report_timer = Instant::now();
        }

        sleep(Duration::from_millis(2)).await;
    }
}

async fn handle_solution(
    sol:              Solution,
    sink:             &SolutionSink,
    file_loader:      &Arc<Option<kaspa_pow::genome_file::FileGenomeLoader>>,
    packed_bytes:     &Arc<Vec<u8>>,
    frag_size:        u32,
    dash:             &std::sync::Arc<std::sync::Mutex<DashStats>>,
) {
    let Solution { nonce, template, gpu_id } = sol;

    // ── Genome PoW CPU cross-check (works in both node and stratum mode) ──
    if template.genome_active {
        let cpu_pow = genome_mix_hash(packed_bytes, &template.epoch_seed, nonce, &template.pre_pow_hash);
        let state   = GenomePowState::new(template.pre_pow_hash, template.target, template.epoch_seed, frag_size);
        if cpu_pow > state.target {
            warn!("[GPU{}] Genome PoW false-positive nonce={:#018x} — skipping", gpu_id, nonce);
            return;
        }
        let frag_idx = fragment_index(&template.epoch_seed, nonce, frag_size);
        let synth    = SyntheticLoader::new(frag_size, template.epoch_seed);
        let fl: &dyn GenomeDatasetLoader = match file_loader.as_ref() {
            Some(f) => f,
            None    => &synth,
        };
        let fragment  = fl.load_fragment(frag_idx).unwrap_or_else(|| vec![0u8; frag_size as usize]);
        let (_, _, fitness) = state.check_pow_with_fragment(nonce, &fragment);
        info!("[GPU{}] Genome PoW PASS nonce={:#018x} fitness={}", gpu_id, nonce, fitness);
    }

    // ── Submit via node gRPC or stratum pool ──
    match sink {
        SolutionSink::Node(rpc) => {
            let rpc_block = match &template.source {
                MiningSource::Node { rpc_block, .. } => rpc_block.clone(),
                MiningSource::Stratum { .. } => {
                    warn!("[GPU{gpu_id}] Node sink with stratum template — cannot submit");
                    return;
                }
            };
            match rpc.submit_block(build_raw_block_nonce(&rpc_block, nonce), false).await {
                Ok(r) => {
                    let is_ibd = matches!(r.report, SubmitBlockReport::Reject(SubmitBlockRejectReason::IsInIBD));
                    info!("[GPU{}] submit_block: {:?}", gpu_id, r.report);
                    let accepted = matches!(r.report, SubmitBlockReport::Success);
                    {
                        let mut s = dash.lock().unwrap();
                        let g_idx = s.gpus.iter().position(|g| g.id == gpu_id);
                        if accepted {
                            s.accepted += 1;
                            if let Some(i) = g_idx { s.gpus[i].accepted += 1; }
                            s.push_log(format!("[GPU{gpu_id}] Block accepted  daa={}", template.daa_score));
                        } else {
                            s.rejected += 1;
                            if let Some(i) = g_idx { s.gpus[i].rejected += 1; }
                            s.push_log(format!("[GPU{gpu_id}] Block rejected  {:?}", r.report));
                        }
                    }
                    if is_ibd { warn!("Node still in IBD"); }
                }
                Err(e) => {
                    warn!("[GPU{}] submit_block error: {e}", gpu_id);
                    dash.lock().unwrap().push_log(format!("[GPU{gpu_id}] submit error: {e}"));
                }
            }
        }
        SolutionSink::Stratum(sol_tx) => {
            let job_id = match &template.source {
                MiningSource::Stratum { job_id, .. } => job_id.clone(),
                MiningSource::Node { .. } => {
                    warn!("[GPU{gpu_id}] Stratum sink with node template — cannot submit");
                    return;
                }
            };
            let extranonce2 = (nonce & 0xFFFF_FFFF) as u32;
            if sol_tx.send(crate::stratum_client::StratumSolution { job_id, extranonce2 }).await.is_err() {
                warn!("[GPU{gpu_id}] stratum solution channel closed");
            }
        }
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
