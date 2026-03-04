use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use clap::ArgMatches;
use kaspa_addresses::Address;
use kaspa_consensus_core::header::Header;
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_pow::{genome_pow::{
    fragment_index, genome_mix_hash, GenomeDatasetLoader,
    GenomePowState, SyntheticLoader, GENOME_BASE_SIZE, MIX_CHUNK_BYTES,
}, matrix::Matrix, State as KHeavyState};
use kaspa_rpc_core::{api::rpc::RpcApi, model::message::GetBlockTemplateRequest, RpcRawBlock,
    SubmitBlockReport, SubmitBlockRejectReason};
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

impl GpuContext {
    async fn new() -> Self {
        // Allow all real GPU vendors: NVIDIA (Vulkan/DX12), AMD (Vulkan/DX12),
        // Intel (Vulkan/DX12), Apple (Metal).
        // Software renderers (llvmpipe, lavapipe) are rejected via DeviceType::Cpu check.
        let all_backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: all_backends,
            ..Default::default()
        });

        // Prefer discrete GPU over integrated; skip CPU/software renderers entirely.
        let adapter = {
            let mut candidates: Vec<wgpu::Adapter> = instance
                .enumerate_adapters(all_backends)
                .into_iter()
                .filter(|a| {
                    let info = a.get_info();
                    info.device_type != wgpu::DeviceType::Cpu
                        && !info.name.to_lowercase().contains("llvmpipe")
                        && !info.name.to_lowercase().contains("lavapipe")
                        && !info.name.to_lowercase().contains("softpipe")
                })
                .collect();
            // Sort: discrete first, then integrated, then other
            candidates.sort_by_key(|a| match a.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu   => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                _                               => 2,
            });
            candidates.into_iter().next()
                .expect("No real GPU adapter found. genome-miner requires Metal, Vulkan, or DX12 (NVIDIA / AMD / Intel / Apple).")
        };

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
}

// ── Packed genome helpers ────────────────────────────────────────

/// Build a synthetic packed genome for devnet/testing (no real file).
/// Not consensus-relevant — only used when genome PoW is active at daa_score=0 on devnet.
fn synthetic_packed_genome(frag_size: u32) -> Vec<u8> {
    let num_fragments = (GENOME_BASE_SIZE / frag_size.max(1) as u64) as usize;
    let packed_frag = frag_size as usize / 4;
    // Deterministic repeating pattern per fragment index.
    (0..num_fragments).flat_map(|i| std::iter::repeat((i & 0xFF) as u8).take(packed_frag)).collect()
}

// ── KHeavyHash matrix helpers ────────────────────────────────────────────────

/// Generate the KHeavyHash matrix from pre_pow_hash and return as raw bytes
/// (4096 × u32 LE, row-major, values 0-15).  ~16 KB, uploaded once per template.
fn build_matrix_bytes(pre_pow_hash: &kaspa_hashes::Hash) -> Vec<u8> {
    let matrix = Matrix::generate(*pre_pow_hash);
    let flat = matrix.to_flat_u32();
    flat.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Build the 88-byte KHeavyParams buffer for the WGSL KHeavyHash shader.
fn build_kheavy_params(
    pre_pow_hash: &kaspa_hashes::Hash,
    timestamp: u64,
    target: &kaspa_math::Uint256,
    nonce_base: u64,
) -> Vec<u8> {
    let mut buf = vec![0u8; 88];
    buf[0..32].copy_from_slice(pre_pow_hash.as_ref());
    buf[32..36].copy_from_slice(&(timestamp as u32).to_le_bytes());
    buf[36..40].copy_from_slice(&((timestamp >> 32) as u32).to_le_bytes());
    buf[40..72].copy_from_slice(&target.to_le_bytes());
    buf[72..76].copy_from_slice(&(nonce_base as u32).to_le_bytes());
    buf[76..80].copy_from_slice(&((nonce_base >> 32) as u32).to_le_bytes());
    // buf[80..88] = 0 (pad)
    buf
}

/// GPU KHeavyHash dispatch.  matrix_buf holds the pre-uploaded 64×64 matrix.
/// Returns `Some((nonce, gpu_hash))` on success so the caller can CPU-verify the hash.
async fn gpu_search_kheavy(
    ctx: &GpuContext,
    params_data: &[u8],
    matrix_buf: &wgpu::Buffer,
    batch_size: u32,
) -> Option<(u64, [u32; 8])> {
    let dev = &ctx.device;

    let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("kh_params"),
        contents: params_data,
        usage:    wgpu::BufferUsages::STORAGE,
    });

    // 48 bytes: found(4) + nonce_lo(4) + nonce_hi(4) + pad(4) + dbg_hash(32)
    let output_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("kh_output"),
        size:               48,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    output_buf.slice(..).get_mapped_range_mut().fill(0);
    output_buf.unmap();

    let readback_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("kh_readback"),
        size:               48,
        usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label:  Some("kh_bg"),
        layout: &ctx.bind_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: matrix_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&ctx.kh_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, 48);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let slice = readback_buf.slice(..);
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
        gpu_hash[i] = u32::from_le_bytes(data[16 + i*4..16 + i*4 + 4].try_into().unwrap());
    }
    drop(data);
    readback_buf.unmap();

    if found != 0 {
        Some(((nonce_lo as u64) | ((nonce_hi as u64) << 32), gpu_hash))
    } else {
        None
    }
}

// ── GPU nonce batch ───────────────────────────────────────────────────────────

/// One GPU dispatch: nonce_base .. nonce_base+batch_size.
/// Returns the winning nonce or None.
async fn gpu_search_batch(
    ctx: &GpuContext,
    params_data: &[u8],       // 112-byte Params struct (see WGSL)
    frag_hash_buf: &wgpu::Buffer,
    batch_size: u32,
) -> Option<u64> {
    let dev = &ctx.device;

    // Params uniform buffer (updated each batch)
    let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("params"),
        contents: params_data,
        usage:    wgpu::BufferUsages::STORAGE,
    });

    // Output buffer: [found(u32), nonce_lo(u32), nonce_hi(u32), pad(u32)] = 16 bytes
    let output_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("output"),
        size:               16,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    output_buf.slice(..).get_mapped_range_mut().fill(0);
    output_buf.unmap();

    // Readback staging buffer
    let readback_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("readback"),
        size:               16,
        usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label:  Some("bg"),
        layout: &ctx.bind_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: frag_hash_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // workgroup_size=256, dispatch ceil(batch_size/256) groups
        pass.dispatch_workgroups((batch_size + 255) / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, 16);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // Map readback buffer
    let slice = readback_buf.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    dev.poll(wgpu::Maintain::Wait);
    rx.await.ok()?.ok()?;

    let data = slice.get_mapped_range();
    let found    = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let nonce_lo = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let nonce_hi = u32::from_le_bytes(data[8..12].try_into().unwrap());
    drop(data);
    readback_buf.unmap();

    if found != 0 {
        Some((nonce_lo as u64) | ((nonce_hi as u64) << 32))
    } else {
        None
    }
}

// ── Build Params struct bytes ─────────────────────────────────────────────────

#[allow(dead_code)]
fn build_params(
    epoch_seed: &kaspa_hashes::Hash,
    pre_pow_hash: &kaspa_hashes::Hash,
    target: &kaspa_math::Uint256,
    nonce_base: u64,
    num_fragments: u32,
) -> [u8; 64] {
    let mut buf = [0u8; 64];
    buf[0..32].copy_from_slice(epoch_seed.as_ref());
    buf[32..64].copy_from_slice(pre_pow_hash.as_ref());
    // Note: WGSL Params layout:
    //   epoch_seed[32], pre_pow_hash[32] → 64 bytes
    //   target[32], nonce_base_lo(4), nonce_base_hi(4), num_fragments(4), pad(4) → 48 bytes
    // We need a 112-byte struct actually. Let me use bytemuck-style manual packing.
    // This is handled in build_params_full below.
    buf
}

/// Full params buffer: 112 bytes matching the WGSL Params struct.
fn build_params_full(
    epoch_seed: &kaspa_hashes::Hash,
    pre_pow_hash: &kaspa_hashes::Hash,
    target: &kaspa_math::Uint256,
    nonce_base: u64,
    num_mix_chunks: u32,
) -> Vec<u8> {
    let mut buf = vec![0u8; 112];
    buf[0..32].copy_from_slice(epoch_seed.as_ref());
    buf[32..64].copy_from_slice(pre_pow_hash.as_ref());
    buf[64..96].copy_from_slice(&target.to_le_bytes());
    buf[96..100].copy_from_slice(&(nonce_base as u32).to_le_bytes());
    buf[100..104].copy_from_slice(&((nonce_base >> 32) as u32).to_le_bytes());
    buf[104..108].copy_from_slice(&num_mix_chunks.to_le_bytes());
    // buf[108..112] = 0 (pad)
    buf
}

// ── `gpu` subcommand entry point ──────────────────────────────────────────────

pub async fn cmd_gpu(m: &ArgMatches) {
    let rpcserver  = m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:36669".to_owned());
    let addr_str   = m.get_one::<String>("mining-address").cloned().expect("--mining-address required");
    let batch_size = m.get_one::<u32>("batch-size").copied().unwrap_or(1 << 20); // 1M nonces/dispatch
    let frag_size  = m.get_one::<u32>("genome-fragment-size").copied().unwrap_or(1_048_576);
    let genome_activation = crate::resolve_activation(m);
    // Resolve genome file: explicit --genome-file flag → auto-discover ~/.rusty-xenom/grch38.xenom → None
    let genome_path: Option<String> = m.get_one::<String>("genome-file").cloned().or_else(|| {
        let default = dirs::home_dir()?.join(".rusty-xenom").join("grch38.xenom");
        if default.exists() { Some(default.to_string_lossy().into_owned()) } else { None }
    });

    info!("Initialising GPU ...");
    let ctx = Arc::new(GpuContext::new().await);

    let url = format!("grpc://{rpcserver}");
    info!("Connecting to {url}");
    let rpc = Arc::new(GrpcClient::connect(url).await.expect("Failed to connect"));

    let pay_address: kaspa_rpc_core::RpcAddress =
        Address::try_from(addr_str.as_str()).expect("Invalid --mining-address");

    let epoch_seed_zero = kaspa_hashes::Hash::from_bytes([0u8; 32]);

    // Open FileGenomeLoader if --genome-file was provided.
    let file_loader: Option<kaspa_pow::genome_file::FileGenomeLoader> =
        genome_path.as_deref().map(|path| {
            kaspa_pow::genome_file::FileGenomeLoader::open(
                std::path::Path::new(path), frag_size, false,
            )
            .unwrap_or_else(|e| panic!("Failed to open genome file '{path}': {e}"))
        });

    // Synthetic fallback for devnet (only allocated if no file loader).
    let synthetic_bytes: Vec<u8> = if file_loader.is_none() {
        synthetic_packed_genome(frag_size)
    } else {
        Vec::new()
    };

    let (packed_genome_ref, num_mix_chunks): (&[u8], u32) = match file_loader.as_ref() {
        Some(loader) => {
            let packed = loader.packed_dataset().unwrap();
            let chunks = (packed.len() / MIX_CHUNK_BYTES) as u32;
            info!("Genome PoW: loaded {} ({} MB, {chunks} mix-chunks)",
                genome_path.as_deref().unwrap_or(""), packed.len() / 1_048_576);
            (packed, chunks)
        }
        None => {
            let chunks = (synthetic_bytes.len() / MIX_CHUNK_BYTES) as u32;
            info!("Genome PoW: synthetic dataset ({chunks} mix-chunks) — devnet/testing only");
            (&synthetic_bytes, chunks)
        }
    };

    // Upload packed genome to GPU VRAM — uploaded ONCE, never changes between epochs.
    let packed_genome_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("packed_genome"),
        contents: packed_genome_ref,
        usage:    wgpu::BufferUsages::STORAGE,
    });

    // KHeavyHash matrix buffer: 4096 × u32 = 16 KB, updated per template
    let matrix_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("kh_matrix"),
        size:               4096 * 4,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut last_template_id: Option<kaspa_hashes::Hash> = None;
    let mut last_pre_pow_hash: Option<kaspa_hashes::Hash> = None;
    let mut nonce_base: u64 = 0;
    let mut total_hashes: u64 = 0;
    let mut report_timer = Instant::now();

    loop {
        // Fetch template
        let resp = match rpc.get_block_template_call(None, GetBlockTemplateRequest::new(pay_address.clone(), vec![])).await {
            Ok(r) => r,
            Err(e) => { warn!("get_block_template: {e}"); sleep(Duration::from_secs(1)).await; continue; }
        };
        let rpc_block: RpcRawBlock = resp.block;
        if !resp.is_synced {
            warn!("Node not synced — waiting for IBD to complete");
            sleep(Duration::from_secs(2)).await;
            continue;
        }

        let current_id = rpc_block.header.accepted_id_merkle_root;
        if last_template_id == Some(current_id) {
            sleep(Duration::from_millis(50)).await;
        } else {
            last_template_id = Some(current_id);
            nonce_base = 0;
            info!("New template daa={}", rpc_block.header.daa_score);
        }

        let header: Header = (&rpc_block.header).into();

        // epoch_seed is included in the Params buffer per-batch — no VRAM update needed.

        if header.daa_score < genome_activation {
            // Pre-activation: mine KHeavyHash (PyrinHashv2) on GPU.
            // epoch_seed is included when non-zero (activation=0), matching mainnet block semantics.
            let pre_pow_hash = kaspa_consensus_core::hashing::header::hash_override_nonce_time(&header, 0, 0);
            let target = kaspa_math::Uint256::from_compact_target_bits(header.bits);

            // Re-upload matrix when template changes (pre_pow_hash → new matrix)
            if last_pre_pow_hash != Some(pre_pow_hash) {
                let mat_bytes = build_matrix_bytes(&pre_pow_hash);
                ctx.queue.write_buffer(&matrix_buf, 0, &mat_bytes);
                last_pre_pow_hash = Some(pre_pow_hash);
            }

            let kh_params = build_kheavy_params(&pre_pow_hash, header.timestamp, &target, nonce_base);
            let solution = gpu_search_kheavy(&ctx, &kh_params, &matrix_buf, batch_size).await;
            total_hashes += batch_size as u64;
            nonce_base = nonce_base.wrapping_add(batch_size as u64);

            if let Some((nonce, gpu_hash)) = solution {
                // CPU cross-check: verify the GPU nonce before submitting
                let pow_state = KHeavyState::new(&header);
                let (cpu_valid, cpu_pow) = pow_state.check_pow(nonce);
                // Convert CPU Uint256 → [u32; 8] LE for comparison
                let cpu_bytes = cpu_pow.to_le_bytes();
                let mut cpu_hash = [0u32; 8];
                for i in 0..8 {
                    cpu_hash[i] = u32::from_le_bytes(cpu_bytes[i*4..i*4+4].try_into().unwrap());
                }
                if gpu_hash != cpu_hash {
                    warn!(
                        "GPU KHeavyHash hash mismatch nonce={}: gpu={:08x}{:08x} cpu={:08x}{:08x}",
                        nonce,
                        gpu_hash[7], gpu_hash[6],
                        cpu_hash[7], cpu_hash[6]
                    );
                }
                if !cpu_valid {
                    warn!("GPU KHeavyHash false-positive nonce={} — skipping invalid block", nonce);
                    last_template_id = None;
                    continue;
                }
                let solved = build_raw_block_nonce(&rpc_block, nonce);
                let ibd = match rpc.submit_block(solved, false).await {
                    Ok(r) => {
                        let is_ibd = matches!(r.report, SubmitBlockReport::Reject(SubmitBlockRejectReason::IsInIBD));
                        info!("Block submitted (KHeavyHash GPU): {:?}", r.report);
                        is_ibd
                    }
                    Err(e) => { warn!("submit_block: {e}"); false }
                };
                if !ibd { last_template_id = None; }
            }
            if report_timer.elapsed() >= Duration::from_secs(5) {
                let elapsed = report_timer.elapsed().as_secs_f64();
                let mhs = total_hashes as f64 / elapsed / 1_000_000.0;
                info!("GPU [KHeavyHash] [{:.2} MH/s] daa={} (genome activates at {})",
                    mhs, header.daa_score, genome_activation);
                total_hashes = 0;
                report_timer = Instant::now();
            }
            continue;
        }

        let pre_pow_hash = kaspa_consensus_core::hashing::header::hash_override_nonce_time(&header, 0, 0);
        let target = kaspa_math::Uint256::from_compact_target_bits(header.bits);

        // Build params for this batch
        let params_bytes = build_params_full(
            &header.epoch_seed,
            &pre_pow_hash,
            &target,
            nonce_base,
            num_mix_chunks,
        );

        let solution = gpu_search_batch(&ctx, &params_bytes, &packed_genome_buf, batch_size).await;
        total_hashes += batch_size as u64;
        nonce_base = nonce_base.wrapping_add(batch_size as u64);

        if let Some(nonce) = solution {
            // CPU cross-check: re-run genome_mix_hash on CPU before submitting.
            let pre_pow   = kaspa_consensus_core::hashing::header::hash_override_nonce_time(&header, 0, 0);
            let target    = kaspa_math::Uint256::from_compact_target_bits(header.bits);
            let state     = GenomePowState::new(pre_pow, target, header.epoch_seed, frag_size);
            let cpu_pow   = genome_mix_hash(packed_genome_ref, &header.epoch_seed, nonce, &pre_pow);
            if cpu_pow > state.target {
                warn!(
                    "GPU Genome PoW false-positive nonce={:#018x} cpu_pow_msb={:08x} — skipping invalid block",
                    nonce, cpu_pow.to_le_bytes()[31]
                );
                last_template_id = None;
                continue;
            }
            // Compute fitness separately for the coinbase (still needed for block reward)
            let frag_idx = fragment_index(&header.epoch_seed, nonce, frag_size);
            let synthetic_frag_loader = SyntheticLoader::new(frag_size, header.epoch_seed);
            let frag_loader: &dyn GenomeDatasetLoader = match file_loader.as_ref() {
                Some(fl) => fl,
                None => &synthetic_frag_loader,
            };
            let fragment = frag_loader.load_fragment(frag_idx).unwrap_or_else(|| vec![0u8; frag_size as usize]);
            let (_, _cpu_pow2, cpu_fitness) = state.check_pow_with_fragment(nonce, &fragment);
            info!("CPU cross-check PASSED nonce={:#018x} fitness={}", nonce, cpu_fitness);
            let solved = build_raw_block_nonce(&rpc_block, nonce);
            let ibd = match rpc.submit_block(solved, false).await {
                Ok(r) => {
                    let is_ibd = matches!(r.report, SubmitBlockReport::Reject(SubmitBlockRejectReason::IsInIBD));
                    info!("Block submitted (Genome PoW): {:?}", r.report);
                    is_ibd
                }
                Err(e) => { warn!("submit_block: {e}"); false }
            };
            if !ibd { last_template_id = None; }
        }

        if report_timer.elapsed() >= Duration::from_secs(5) {
            let elapsed = report_timer.elapsed().as_secs_f64();
            let mhs = total_hashes as f64 / elapsed / 1_000_000.0;
            info!(
                "GPU [{:.0} MH/s] daa={} epoch_seed={}...",
                mhs,
                header.daa_score,
                &format!("{:?}", header.epoch_seed)[..8],
            );
            total_hashes = 0;
            report_timer = Instant::now();
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
