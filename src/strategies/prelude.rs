//! Common imports and types for all strategy modules
//!
//! This module provides a convenient prelude for all strategy implementations,
//! reducing boilerplate imports and ensuring consistency across strategies.

// Re-export commonly used types from the main crate
pub use crate::{
    HyperliquidBacktest,
    HyperliquidData, 
    HyperliquidCommission,
    backtest::EnhancedReport,
    errors::{HyperliquidBacktestError as HyperliquidError, Result},
};

// Re-export base strategy types
pub use super::base::{
    TradingStrategy,
    StrategyType,
    StrategyConfig,
    CommissionConfig, 
    RiskLevel,
    ExecutionMode,
    StrategyMetrics,
    StrategyFactory,
    StrategyComparison,
};

// Standard library imports commonly used in strategies
pub use std::collections::HashMap;

// External crate imports
pub use chrono::{DateTime, FixedOffset, Utc};
pub use serde::{Serialize, Deserialize};

// Logging
pub use log::{debug, info, warn, error};

// Math utilities
pub use std::f64::consts::PI;
pub use std::cmp::{min, max};

/// Utility functions for strategy development
pub mod utils {
    use super::*;

    /// Calculate simple moving average
    pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
        if values.len() < period {
            return vec![];
        }

        let mut result = Vec::new();
        for i in period..=values.len() {
            let sum: f64 = values[i-period..i].iter().sum();
            result.push(sum / period as f64);
        }
        result
    }

    /// Calculate exponential moving average
    pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(values.len());
        result.push(values[0]);

        for &value in values.iter().skip(1) {
            let prev_ema = *result.last().unwrap();
            result.push(alpha * value + (1.0 - alpha) * prev_ema);
        }

        result
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Calculate Bollinger Bands
    pub fn bollinger_bands(values: &[f64], period: usize, std_mult: f64) -> Vec<(f64, f64, f64)> {
        if values.len() < period {
            return vec![];
        }

        let mut result = Vec::new();
        
        for i in period..=values.len() {
            let slice = &values[i-period..i];
            let ma = slice.iter().sum::<f64>() / period as f64;
            let std = std_dev(slice);
            
            let upper = ma + std_mult * std;
            let lower = ma - std_mult * std;
            
            result.push((upper, ma, lower));
        }
        
        result
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(values: &[f64], period: usize) -> Vec<f64> {
        if values.len() < period + 1 {
            return vec![];
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        // Calculate price changes
        for i in 1..values.len() {
            let change = values[i] - values[i-1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let mut result = Vec::new();
        
        // Calculate RSI for each period
        for i in period..=gains.len() {
            let avg_gain = gains[i-period..i].iter().sum::<f64>() / period as f64;
            let avg_loss = losses[i-period..i].iter().sum::<f64>() / period as f64;
            
            if avg_loss == 0.0 {
                result.push(100.0);
            } else {
                let rs = avg_gain / avg_loss;
                let rsi = 100.0 - (100.0 / (1.0 + rs));
                result.push(rsi);
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    pub fn macd(values: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) 
        -> Vec<(f64, f64, f64)> {
        let fast_ema = ema(values, fast_period);
        let slow_ema = ema(values, slow_period);
        
        if fast_ema.len() != slow_ema.len() {
            return vec![];
        }

        let macd_line: Vec<f64> = fast_ema.iter()
            .zip(slow_ema.iter())
            .map(|(fast, slow)| fast - slow)
            .collect();

        let signal_line = ema(&macd_line, signal_period);
        
        let mut result = Vec::new();
        let start_idx = macd_line.len() - signal_line.len();
        
        for (i, &signal) in signal_line.iter().enumerate() {
            let macd_val = macd_line[start_idx + i];
            let histogram = macd_val - signal;
            result.push((macd_val, signal, histogram));
        }

        result
    }

    /// Generate realistic price data for testing
    pub fn generate_realistic_ohlc(
        start_price: f64, 
        num_points: usize, 
        volatility: f64,
        trend: f64
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        
        let mut current_price = start_price;
        
        for i in 0..num_points {
            let open_price = current_price;
            
            // Add trend component
            let trend_component = trend * (i as f64 / num_points as f64);
            
            // Add random walk component
            let random_change = rng.gen_range(-volatility..volatility);
            
            // Calculate close price
            let close_price = current_price + trend_component + random_change;
            
            // Generate high and low based on volatility
            let intrabar_volatility = volatility * 0.5;
            let high_price = f64::max(open_price, close_price) + rng.gen_range(0.0..intrabar_volatility);
            let low_price = f64::min(open_price, close_price) - rng.gen_range(0.0..intrabar_volatility);
            
            // Generate volume
            let base_volume = 1000.0;
            let volume_noise = rng.gen_range(0.5..1.5);
            let vol = base_volume * volume_noise;
            
            open.push(open_price);
            high.push(high_price);
            low.push(low_price);
            close.push(close_price);
            volume.push(vol);
            
            current_price = close_price;
        }
        
        (open, high, low, close, volume)
    }

    /// Calculate correlation between two price series
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma_result = sma(&values, 3);
        assert_eq!(sma_result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema_result = ema(&values, 3);
        assert_eq!(ema_result.len(), 5);
        assert_eq!(ema_result[0], 1.0); // First value
    }

    #[test]
    fn test_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = std_dev(&values);
        assert!(std > 0.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001); // Perfect positive correlation
    }
}
