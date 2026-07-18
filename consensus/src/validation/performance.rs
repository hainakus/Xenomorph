//! Performance optimization and tuning for ZK validation system
//! 
//! This module provides performance monitoring, caching strategies,
//! and adaptive parameter tuning for optimal validation performance.

use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Performance metrics for validation operations
#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    pub validation_time_ms: f64,
    pub selection_time_ms: f64,
    pub signature_aggregation_time_ms: f64,
    pub network_time_ms: f64,
    pub total_time_ms: f64,
    pub memory_usage_bytes: u64,
    pub cpu_usage_percent: f64,
}

impl PerformanceMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            validation_time_ms: 0.0,
            selection_time_ms: 0.0,
            signature_aggregation_time_ms: 0.0,
            network_time_ms: 0.0,
            total_time_ms: 0.0,
            memory_usage_bytes: 0,
            cpu_usage_percent: 0.0,
        }
    }

    /// Calculate p50 (median) from a collection
    pub fn p50(metrics: &[Self]) -> Self {
        if metrics.is_empty() {
            return Self::new();
        }

        let mut sorted: Vec<_> = metrics.iter().collect();
        sorted.sort_by(|a, b| a.total_time_ms.partial_cmp(&b.total_time_ms).unwrap());

        let idx = sorted.len() / 2;
        sorted[idx].clone()
    }

    /// Calculate p95 from a collection
    pub fn p95(metrics: &[Self]) -> Self {
        if metrics.is_empty() {
            return Self::new();
        }

        let mut sorted: Vec<_> = metrics.iter().collect();
        sorted.sort_by(|a, b| a.total_time_ms.partial_cmp(&b.total_time_ms).unwrap());

        let idx = (sorted.len() as f64 * 0.95) as usize;
        sorted[idx.min(sorted.len() - 1)].clone()
    }

    /// Calculate p99 from a collection
    pub fn p99(metrics: &[Self]) -> Self {
        if metrics.is_empty() {
            return Self::new();
        }

        let mut sorted: Vec<_> = metrics.iter().collect();
        sorted.sort_by(|a, b| a.total_time_ms.partial_cmp(&b.total_time_ms).unwrap());

        let idx = (sorted.len() as f64 * 0.99) as usize;
        sorted[idx.min(sorted.len() - 1)].clone()
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance monitor for tracking validation performance
pub struct PerformanceMonitor {
    metrics_history: VecDeque<PerformanceMetrics>,
    max_history_size: usize,
    current_metrics: PerformanceMetrics,
    operation_start: Option<Instant>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(max_history_size: usize) -> Self {
        Self {
            metrics_history: VecDeque::with_capacity(max_history_size),
            max_history_size,
            current_metrics: PerformanceMetrics::new(),
            operation_start: None,
        }
    }

    /// Start timing an operation
    pub fn start_operation(&mut self) {
        self.operation_start = Some(Instant::now());
        self.current_metrics = PerformanceMetrics::new();
    }

    /// End timing an operation
    pub fn end_operation(&mut self) {
        if let Some(start) = self.operation_start {
            let elapsed = start.elapsed();
            self.current_metrics.total_time_ms = elapsed.as_millis() as f64;
            self.record_metrics();
        }
        self.operation_start = None;
    }

    /// Record validation time
    pub fn record_validation_time(&mut self, time_ms: f64) {
        self.current_metrics.validation_time_ms = time_ms;
    }

    /// Record selection time
    pub fn record_selection_time(&mut self, time_ms: f64) {
        self.current_metrics.selection_time_ms = time_ms;
    }

    /// Record signature aggregation time
    pub fn record_signature_aggregation_time(&mut self, time_ms: f64) {
        self.current_metrics.signature_aggregation_time_ms = time_ms;
    }

    /// Record network time
    pub fn record_network_time(&mut self, time_ms: f64) {
        self.current_metrics.network_time_ms = time_ms;
    }

    /// Record current metrics to history
    fn record_metrics(&mut self) {
        self.metrics_history.push_back(self.current_metrics.clone());
        
        if self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> Vec<PerformanceMetrics> {
        self.metrics_history.iter().cloned().collect()
    }

    /// Get average metrics
    pub fn get_average_metrics(&self) -> PerformanceMetrics {
        if self.metrics_history.is_empty() {
            return PerformanceMetrics::new();
        }

        let count = self.metrics_history.len() as f64;
        let mut avg = PerformanceMetrics::new();

        for metrics in &self.metrics_history {
            avg.validation_time_ms += metrics.validation_time_ms;
            avg.selection_time_ms += metrics.selection_time_ms;
            avg.signature_aggregation_time_ms += metrics.signature_aggregation_time_ms;
            avg.network_time_ms += metrics.network_time_ms;
            avg.total_time_ms += metrics.total_time_ms;
        }

        avg.validation_time_ms /= count;
        avg.selection_time_ms /= count;
        avg.signature_aggregation_time_ms /= count;
        avg.network_time_ms /= count;
        avg.total_time_ms /= count;

        avg
    }

    /// Get recent performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        let metrics: Vec<_> = self.metrics_history.iter().cloned().collect();
        
        if metrics.is_empty() {
            return PerformanceStats::default();
        }

        PerformanceStats {
            average: self.get_average_metrics(),
            p50: PerformanceMetrics::p50(&metrics),
            p95: PerformanceMetrics::p95(&metrics),
            p99: PerformanceMetrics::p99(&metrics),
            sample_count: metrics.len(),
        }
    }
}

/// Performance statistics
#[derive(Clone, Debug)]
pub struct PerformanceStats {
    pub average: PerformanceMetrics,
    pub p50: PerformanceMetrics,
    pub p95: PerformanceMetrics,
    pub p99: PerformanceMetrics,
    pub sample_count: usize,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            average: PerformanceMetrics::new(),
            p50: PerformanceMetrics::new(),
            p95: PerformanceMetrics::new(),
            p99: PerformanceMetrics::new(),
            sample_count: 0,
        }
    }
}

/// Adaptive parameter tuner for optimizing validation parameters
pub struct AdaptiveTuner {
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    tuning_interval: Duration,
    last_tuning: Instant,
}

impl AdaptiveTuner {
    /// Create a new adaptive tuner
    pub fn new(performance_monitor: Arc<Mutex<PerformanceMonitor>>, tuning_interval: Duration) -> Self {
        Self {
            performance_monitor,
            tuning_interval,
            last_tuning: Instant::now(),
        }
    }

    /// Check if tuning is needed
    pub fn should_tune(&self) -> bool {
        self.last_tuning.elapsed() >= self.tuning_interval
    }

    /// Tune parameters based on performance
    pub fn tune_parameters(&mut self) -> TuningRecommendations {
        let monitor = self.performance_monitor.lock().unwrap();
        let stats = monitor.get_performance_stats();
        drop(monitor);

        let mut recommendations = TuningRecommendations::default();

        // Analyze validation time
        if stats.p95.validation_time_ms > 100.0 {
            recommendations.reduce_sample_size = true;
            recommendations.increase_timeout = true;
        }

        // Analyze network time
        if stats.p95.network_time_ms > 1000.0 {
            recommendations.increase_fanout = true;
            recommendations.reduce_timeout = true;
        }

        // Analyze signature aggregation time
        if stats.p95.signature_aggregation_time_ms > 50.0 {
            recommendations.batch_aggregation = true;
        }

        self.last_tuning = Instant::now();
        recommendations
    }
}

/// Tuning recommendations
#[derive(Clone, Debug, Default)]
pub struct TuningRecommendations {
    pub reduce_sample_size: bool,
    pub increase_sample_size: bool,
    pub increase_timeout: bool,
    pub reduce_timeout: bool,
    pub increase_fanout: bool,
    pub reduce_fanout: bool,
    pub batch_aggregation: bool,
    pub enable_caching: bool,
    pub disable_caching: bool,
}

/// LRU cache for validation results
pub struct ValidationCache {
    cache: lru::LruCache<String, CachedValidationResult>,
    max_size: usize,
    hit_count: u64,
    miss_count: u64,
}

#[derive(Clone)]
struct CachedValidationResult {
    result: bool,
    timestamp: Instant,
}

impl ValidationCache {
    /// Create a new validation cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: lru::LruCache::new(max_size),
            max_size,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Get a cached validation result
    pub fn get(&mut self, key: &str) -> Option<bool> {
        if let Some(cached) = self.cache.get(key) {
            // Check if cache entry is not too old (5 minutes)
            if cached.timestamp.elapsed() < Duration::from_secs(300) {
                self.hit_count += 1;
                return Some(cached.result);
            }
        }
        
        self.miss_count += 1;
        None
    }

    /// Put a validation result in the cache
    pub fn put(&mut self, key: String, result: bool) {
        let cached = CachedValidationResult {
            result,
            timestamp: Instant::now(),
        };
        self.cache.put(key, cached);
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            return 0.0;
        }
        self.hit_count as f64 / total as f64
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hit_count = 0;
        self.miss_count = 0;
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

/// Batch processor for optimizing bulk operations
pub struct BatchProcessor<T> {
    batch_size: usize,
    timeout: Duration,
    current_batch: Vec<T>,
    batch_start: Instant,
}

impl<T: Clone> BatchProcessor<T> {
    /// Create a new batch processor
    pub fn new(batch_size: usize, timeout: Duration) -> Self {
        Self {
            batch_size,
            timeout,
            current_batch: Vec::new(),
            batch_start: Instant::now(),
        }
    }

    /// Add an item to the batch
    pub fn add(&mut self, item: T) -> Option<Vec<T>> {
        self.current_batch.push(item);
        
        if self.current_batch.len() >= self.batch_size {
            self.flush()
        } else if self.batch_start.elapsed() >= self.timeout {
            self.flush()
        } else {
            None
        }
    }

    /// Flush the current batch
    pub fn flush(&mut self) -> Option<Vec<T>> {
        if self.current_batch.is_empty() {
            return None;
        }
        
        let batch = std::mem::take(&mut self.current_batch);
        self.batch_start = Instant::now();
        Some(batch)
    }

    /// Get current batch size
    pub fn current_size(&self) -> usize {
        self.current_batch.len()
    }
}

/// Performance profiler for detailed analysis
pub struct PerformanceProfiler {
    enabled: bool,
    profile_data: Vec<ProfileEntry>,
}

#[derive(Clone, Debug)]
struct ProfileEntry {
    operation: String,
    duration: Duration,
    timestamp: Instant,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            profile_data: Vec::new(),
        }
    }

    /// Profile an operation
    pub fn profile<F, R>(&mut self, operation: String, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.profile_data.push(ProfileEntry {
            operation,
            duration,
            timestamp: Instant::now(),
        });

        result
    }

    /// Get profile data
    pub fn get_profile_data(&self) -> Vec<ProfileEntry> {
        self.profile_data.clone()
    }

    /// Clear profile data
    pub fn clear(&mut self) {
        self.profile_data.clear();
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        if self.profile_data.is_empty() {
            return "No profile data available".to_string();
        }

        let mut report = String::new();
        report.push_str("Performance Profile Report:\n");
        report.push_str("=========================\n");

        for entry in &self.profile_data {
            report.push_str(&format!(
                "{}: {:.2}ms\n",
                entry.operation,
                entry.duration.as_millis() as f64
            ));
        }

        let total: Duration = self.profile_data.iter().map(|e| e.duration).sum();
        report.push_str(&format!("Total: {:.2}ms\n", total.as_millis() as f64));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.validation_time_ms = 50.0;
        metrics.selection_time_ms = 10.0;
        metrics.total_time_ms = 60.0;

        assert_eq!(metrics.validation_time_ms, 50.0);
        assert_eq!(metrics.total_time_ms, 60.0);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new(100);
        
        monitor.start_operation();
        monitor.record_validation_time(50.0);
        monitor.end_operation();

        assert_eq!(monitor.get_metrics_history().len(), 1);
    }

    #[test]
    fn test_performance_stats() {
        let mut monitor = PerformanceMonitor::new(100);
        
        for i in 0..10 {
            monitor.start_operation();
            monitor.record_validation_time(i as f64);
            monitor.end_operation();
        }

        let stats = monitor.get_performance_stats();
        assert_eq!(stats.sample_count, 10);
        assert!(stats.average.validation_time_ms > 0.0);
    }

    #[test]
    fn test_validation_cache() {
        let mut cache = ValidationCache::new(100);
        
        cache.put("key1".to_string(), true);
        assert_eq!(cache.get("key1"), Some(true));
        assert_eq!(cache.get("key2"), None);
        
        assert_eq!(cache.hit_rate(), 0.5);
    }

    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::new(3, Duration::from_secs(1));
        
        assert!(processor.add(1).is_none());
        assert!(processor.add(2).is_none());
        
        let batch = processor.add(3).unwrap();
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new(true);
        
        let result = profiler.profile("test_op".to_string(), || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert_eq!(profiler.get_profile_data().len(), 1);
    }

    #[test]
    fn test_adaptive_tuner() {
        let monitor = Arc::new(Mutex::new(PerformanceMonitor::new(100)));
        let mut tuner = AdaptiveTuner::new(monitor, Duration::from_secs(60));
        
        assert!(!tuner.should_tune());
        
        let recommendations = tuner.tune_parameters();
        // Should return recommendations even with no data
        assert!(!recommendations.reduce_sample_size);
    }
}
