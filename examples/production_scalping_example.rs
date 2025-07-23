//! Production Scalping Strategy Example
//!
//! This example demonstrates how to use the production-ready scalping strategy
//! from the strategies module instead of the basic examples.

use hyperliquid::prelude::*;
use hyperliquid::strategies::scalping::ScalpingConfig;
use hyperliquid::strategies::{StrategyComparison, StrategyFactory};

fn main() -> hyperliquid::Result<()> {
    // Initialize logging
    println!("🚀 Production Scalping Strategy");
    println!("================================\n");

    // Create multiple scalping strategies with different configurations
    let strategies = vec![
        // Default configuration (EMA 3/8)
        StrategyFactory::create_scalping(),
        // Faster scalping (EMA 2/5)
        StrategyFactory::create_scalping_with_config(ScalpingConfig {
            fast_period: 2,
            slow_period: 5,
            initial_capital: 100.0,
            maker_rate: 0.0001,
            taker_rate: 0.0003,
            timeframe: "1m".to_string(),
            data_hours: 2,
            ..Default::default()
        }),
        // More conservative scalping (EMA 5/12)
        StrategyFactory::create_scalping_with_config(ScalpingConfig {
            fast_period: 5,
            slow_period: 12,
            initial_capital: 100.0,
            maker_rate: 0.0001,
            taker_rate: 0.0003,
            timeframe: "5m".to_string(),
            data_hours: 6,
            ..Default::default()
        }),
    ];

    // Generate sample data for comparison
    let sample_data = generate_sample_btc_data()?;

    // Compare strategies
    println!("Running strategy comparison...\n");
    let comparison_results = StrategyComparison::compare_strategies(strategies, &sample_data)?;

    // Print comparison results
    StrategyComparison::print_comparison(&comparison_results);

    // Run detailed analysis on the best performing strategy
    println!("\n📋 Detailed Analysis of Default Scalping Strategy");
    println!("================================================\n");

    let scalping_strategy = hyperliquid::strategies::scalping::ScalpingStrategy::new();
    let detailed_results = scalping_strategy.run_with_sample_data()?;

    // Print comprehensive results
    detailed_results.print_results();

    // Export results to CSV
    scalping_strategy.export_results(&detailed_results, "production_scalping_results.csv")?;
    println!("✅ Results exported to production_scalping_results.csv");

    Ok(())
}

/// Generate sample BTC data for strategy comparison
fn generate_sample_btc_data() -> hyperliquid::Result<HyperliquidData> {
    use chrono::Utc;

    let mut datetime = Vec::new();
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();
    let mut funding_rates = Vec::new();

    let start_time = Utc::now() - chrono::Duration::hours(6);
    let mut base_price = 43000.0;

    // Generate 6 hours of 5-minute data (72 points)
    for i in 0..72 {
        let timestamp = start_time + chrono::Duration::minutes(i * 5);
        datetime.push(timestamp.with_timezone(&chrono::FixedOffset::east_opt(0).unwrap()));

        // Create realistic price movement with trend and noise
        let trend = (i as f64 / 72.0) * 50.0; // Slight upward trend
        let wave = ((i as f64 / 6.0) * 2.0 * std::f64::consts::PI).sin() * 15.0;
        let noise = ((i as f64 * 1.1).sin() + (i as f64 * 0.7).cos()) * 5.0;

        let total_move = trend + wave + noise;
        base_price += total_move;

        let open_price = base_price;
        let close_price = base_price + ((i as f64 % 3.0) - 1.0) * 8.0;
        let high_price = f64::max(open_price, close_price) + 12.0;
        let low_price = f64::min(open_price, close_price) - 12.0;

        open.push(open_price);
        high.push(high_price);
        low.push(low_price);
        close.push(close_price);
        volume.push(1000.0 + (i as f64 * 10.0));
        funding_rates.push(0.0001); // 0.01% funding rate

        base_price = close_price;
    }

    HyperliquidData::with_ohlc_and_funding_data(
        "BTC".to_string(),
        datetime,
        open,
        high,
        low,
        close,
        volume,
        funding_rates,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_generation() {
        let data = generate_sample_btc_data().unwrap();
        assert!(!data.symbol().is_empty());
        assert!(data.len() > 0);
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = StrategyFactory::create_scalping();
        assert!(strategy.name().contains("Scalping"));
    }
}
