use anyhow::Result;

#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// GPU compute utilization percentage (0-100).
    pub utilization: u32,
    /// GPU memory currently in use, in bytes.
    pub memory_used: u64,
    /// GPU temperature in degrees Celsius.
    pub temperature: u32,
}

/// Monitor GPU metrics for the currently active training device.
pub struct GpuMonitor;

impl GpuMonitor {
    /// Query NVIDIA GPU stats via NVML. Only available when the `cuda` feature is enabled.
    #[cfg(feature = "cuda")]
    pub fn get_stats(device_index: u32) -> Result<GpuStats> {
        use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init()?;
        let device = nvml.device_by_index(device_index)?;
        let util = device.utilization_rates()?;
        let mem = device.memory_info()?;
        let temp = device.temperature(TemperatureSensor::Gpu)?;

        Ok(GpuStats {
            utilization: util.gpu,
            memory_used: mem.used,
            temperature: temp,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn get_stats(_device_index: u32) -> Result<GpuStats> {
        anyhow::bail!("GPU monitoring requires the 'cuda' feature")
    }
}
