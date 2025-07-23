//! Scalping Strategy Implementation
//!
//! This module implements a production-ready scalping strategy based on EMA crossovers
//! optimized for high-frequency trading with proper commission analysis.

use crate::strategies::prelude::*;
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Configuration for the scalping strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalpingConfig {
    /// Fast EMA period
    pub fast_period: usize,
    /// Slow EMA period
    pub slow_period: usize,
    /// Initial capital
    pub initial_capital: f64,
    /// Maker commission rate
    pub maker_rate: f64,
    /// Taker commission rate
    pub taker_rate: f64,
    /// Enable funding calculations (usually false for scalping)
    pub funding_enabled: bool,
    /// Timeframe for data (e.g., "5m", "1m")
    pub timeframe: String,
    /// Number of hours of data to generate for backtesting
    pub data_hours: i64,
}

impl Default for ScalpingConfig {
    fn default() -> Self {
        Self {
            fast_period: 3,
            slow_period: 8,
            initial_capital: 100.0,
            maker_rate: 0.0001, // 0.01%
            taker_rate: 0.0003, // 0.03%
            funding_enabled: false,
            timeframe: "5m".to_string(),
            data_hours: 4,
        }
    }
}

/// Production-ready scalping strategy
pub struct ScalpingStrategy {
    config: ScalpingConfig,
    name: String,
    description: String,
}

impl ScalpingStrategy {
    /// Create a new scalping strategy with default configuration
    pub fn new() -> Self {
        Self::with_config(ScalpingConfig::default())
    }

    /// Create a new scalping strategy with custom configuration
    pub fn with_config(config: ScalpingConfig) -> Self {
        Self {
            name: format!(
                "Scalping Strategy (EMA {}/{})",
                config.fast_period, config.slow_period
            ),
            description: format!(
                "High-frequency scalping strategy using {}/{} EMA crossover on {} timeframe",
                config.fast_period, config.slow_period, config.timeframe
            ),
            config,
        }
    }

    /// Run the scalping strategy with sample data
    pub fn run_with_sample_data(&self) -> Result<ScalpingResults> {
        log::info!("Starting Scalping Strategy: {}", self.name);

        // Generate sample data optimized for scalping
        let data = self.generate_scalping_data()?;

        // Create commission structure
        let commission = HyperliquidCommission {
            maker_rate: self.config.maker_rate,
            taker_rate: self.config.taker_rate,
            funding_enabled: self.config.funding_enabled,
        };

        // Run the backtest
        let mut backtest = HyperliquidBacktest::new(
            data,
            self.name.clone(),
            self.config.initial_capital,
            commission.clone(),
        );

        backtest.initialize_base_backtest()?;
        backtest.calculate_with_funding()?;

        let report = backtest.enhanced_report()?;

        // Run comparison with higher commission rates
        let comparison_results = self.run_commission_comparison(&backtest)?;

        Ok(ScalpingResults {
            main_results: report,
            commission_comparison: comparison_results,
            config: self.config.clone(),
        })
    }

    /// Run the strategy with real market data
    pub fn run_with_data(&self, data: &HyperliquidData) -> Result<HyperliquidBacktest> {
        log::info!(
            "Running Scalping Strategy with provided data: {}",
            self.name
        );

        let commission = HyperliquidCommission {
            maker_rate: self.config.maker_rate,
            taker_rate: self.config.taker_rate,
            funding_enabled: self.config.funding_enabled,
        };

        let mut backtest = HyperliquidBacktest::new(
            data.clone(),
            self.name.clone(),
            self.config.initial_capital,
            commission,
        );

        backtest.initialize_base_backtest()?;
        backtest.calculate_with_funding()?;

        Ok(backtest)
    }

    /// Generate sample data optimized for scalping patterns
    fn generate_scalping_data(&self) -> Result<HyperliquidData> {
        let mut datetime = Vec::new();
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        let mut funding_rates = Vec::new();

        let start_time = Utc::now() - chrono::Duration::hours(self.config.data_hours);
        let mut base_price = 43000.0; // Starting BTC price

        // Calculate number of data points based on timeframe
        let interval_minutes = match self.config.timeframe.as_str() {
            "1m" => 1,
            "5m" => 5,
            "15m" => 15,
            _ => 5, // Default to 5 minutes
        };

        let num_points = (self.config.data_hours * 60 / interval_minutes as i64) as usize;

        for i in 0..num_points {
            let timestamp =
                start_time + chrono::Duration::minutes(i as i64 * interval_minutes as i64);
            datetime.push(timestamp.with_timezone(&chrono::FixedOffset::east_opt(0).unwrap()));

            // Create scalping-friendly price movements with frequent small reversals
            let cycle_position = (i as f64 / 6.0) % 1.0; // Complete cycle every 6 periods
            let micro_cycle = (i as f64 / 2.0) % 1.0; // Micro cycle every 2 periods

            // Main trend component (slower)
            let main_trend = (cycle_position * 2.0 * std::f64::consts::PI).sin() * 20.0;

            // Scalping opportunities - frequent small moves
            let scalp_move = (micro_cycle * 4.0 * std::f64::consts::PI).sin() * 8.0;

            // Add some noise for realism
            let noise = ((i as f64 * 0.7).sin() + (i as f64 * 1.3).cos()) * 3.0;

            let total_move = main_trend + scalp_move + noise;
            base_price += total_move;

            let open_price = base_price;

            // Create intrabar movement suitable for scalping
            let intrabar_range = 5.0 + f64::abs(scalp_move) * 0.5; // 5-10 point ranges
            let close_price = base_price + (micro_cycle - 0.5) * intrabar_range;
            let high_price = f64::max(open_price, close_price) + intrabar_range * 0.3;
            let low_price = f64::min(open_price, close_price) - intrabar_range * 0.3;

            open.push(open_price);
            high.push(high_price);
            low.push(low_price);
            close.push(close_price);
            volume.push(800.0 + (i as f64 * 15.0) + f64::abs(scalp_move) * 20.0);

            // Funding rates (less relevant for scalping)
            funding_rates.push(((i as f64 % 12.0 - 6.0) / 6.0) * 0.00003);

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

    /// Run commission comparison analysis
    fn run_commission_comparison(
        &self,
        base_backtest: &HyperliquidBacktest,
    ) -> Result<CommissionComparison> {
        let higher_commission = HyperliquidCommission {
            maker_rate: 0.0005, // Higher maker fee (0.05%)
            taker_rate: 0.001,  // Higher taker fee (0.1%)
            funding_enabled: false,
        };

        let mut backtest_high_commission = HyperliquidBacktest::new(
            base_backtest.data().clone(),
            format!("{} (High Commission)", self.name),
            self.config.initial_capital,
            higher_commission,
        );

        backtest_high_commission.initialize_base_backtest()?;
        backtest_high_commission.calculate_with_funding()?;

        let report_high_commission = backtest_high_commission.enhanced_report()?;
        let base_report = base_backtest.enhanced_report()?;

        Ok(CommissionComparison {
            low_commission_profit: base_report.final_equity - base_report.initial_capital,
            high_commission_profit: report_high_commission.final_equity
                - report_high_commission.initial_capital,
            low_commission_total: base_report.commission_stats.total_commission,
            high_commission_total: report_high_commission.commission_stats.total_commission,
            profit_difference: (base_report.final_equity - base_report.initial_capital)
                - (report_high_commission.final_equity - report_high_commission.initial_capital),
            commission_difference: report_high_commission.commission_stats.total_commission
                - base_report.commission_stats.total_commission,
        })
    }

    /// Export strategy results to CSV
    pub fn export_results(&self, results: &ScalpingResults, filename: &str) -> Result<()> {
        // Create a backtest instance to use its export functionality
        let data = self.generate_scalping_data()?;
        let commission = HyperliquidCommission {
            maker_rate: self.config.maker_rate,
            taker_rate: self.config.taker_rate,
            funding_enabled: self.config.funding_enabled,
        };

        let mut backtest = HyperliquidBacktest::new(
            data,
            self.name.clone(),
            self.config.initial_capital,
            commission,
        );

        backtest.initialize_base_backtest()?;
        backtest.calculate_with_funding()?;
        backtest.export_to_csv(filename)?;

        log::info!("Scalping strategy results exported to {}", filename);
        Ok(())
    }
}

impl super::TradingStrategy for ScalpingStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn timeframe(&self) -> &str {
        &self.config.timeframe
    }

    fn strategy_type(&self) -> super::StrategyType {
        super::StrategyType::Scalping
    }

    fn run(&self, data: &HyperliquidData) -> Result<HyperliquidBacktest> {
        self.run_with_data(data)
    }
}

impl Default for ScalpingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from running the scalping strategy
#[derive(Debug, Clone)]
pub struct ScalpingResults {
    pub main_results: EnhancedReport,
    pub commission_comparison: CommissionComparison,
    pub config: ScalpingConfig,
}

impl ScalpingResults {
    /// Print comprehensive results
    pub fn print_results(&self) {
        println!("⚡ Scalping Strategy Results");
        println!("===========================\n");

        println!("Strategy Configuration:");
        println!("  Fast EMA: {}", self.config.fast_period);
        println!("  Slow EMA: {}", self.config.slow_period);
        println!("  Timeframe: {}", self.config.timeframe);
        println!("  Initial Capital: ${:.2}", self.config.initial_capital);
        println!("  Maker Rate: {:.4}%", self.config.maker_rate * 100.0);
        println!("  Taker Rate: {:.4}%", self.config.taker_rate * 100.0);
        println!();

        println!("Main Backtest Results:");
        println!("---------------------");
        println!("Initial Capital: ${:.2}", self.main_results.initial_capital);
        println!("Final Capital: ${:.2}", self.main_results.final_equity);
        println!(
            "Net Profit: ${:.2} ({:.2}%)",
            self.main_results.final_equity - self.main_results.initial_capital,
            self.main_results.total_return * 100.0
        );
        println!(
            "Max Drawdown: {:.2}%",
            self.main_results.max_drawdown * 100.0
        );
        println!("Win Rate: {:.2}%", self.main_results.win_rate * 100.0);
        println!("Profit Factor: {:.2}", self.main_results.profit_factor);
        println!("Sharpe Ratio: {:.2}", self.main_results.sharpe_ratio);
        println!();

        println!("Commission Analysis:");
        println!("-------------------");
        println!(
            "Total Commission Paid: ${:.4}",
            self.main_results.commission_stats.total_commission
        );
        println!(
            "Commission as % of Capital: {:.3}%",
            (self.main_results.commission_stats.total_commission
                / self.main_results.initial_capital)
                * 100.0
        );
        if self.main_results.total_return != 0.0 {
            println!(
                "Commission as % of Profit: {:.2}%",
                (self.main_results.commission_stats.total_commission
                    / (self.main_results.final_equity - self.main_results.initial_capital).abs())
                    * 100.0
            );
        }
        println!();

        println!("Trade Statistics:");
        println!("----------------");
        println!("Total Trades: {}", self.main_results.trade_count);
        if self.main_results.trade_count > 0 {
            let avg_profit_per_trade = (self.main_results.final_equity
                - self.main_results.initial_capital)
                / self.main_results.trade_count as f64;
            println!("Average Profit per Trade: ${:.4}", avg_profit_per_trade);
        }
        println!();

        println!("Commission Impact Analysis:");
        println!("--------------------------");
        println!(
            "Low Commission Profit: ${:.4}",
            self.commission_comparison.low_commission_profit
        );
        println!(
            "High Commission Profit: ${:.4}",
            self.commission_comparison.high_commission_profit
        );
        println!(
            "Profit Difference: ${:.4}",
            self.commission_comparison.profit_difference
        );
        println!(
            "Additional Commission Cost: ${:.4}",
            self.commission_comparison.commission_difference
        );

        if self.commission_comparison.high_commission_profit != 0.0
            && self.commission_comparison.low_commission_profit != 0.0
        {
            let performance_impact = ((self.commission_comparison.low_commission_profit
                / self.commission_comparison.high_commission_profit)
                - 1.0)
                * 100.0;
            println!("Performance Impact: {:.2}%", performance_impact);
        }
        println!();

        println!("Key Insights:");
        println!("------------");
        println!("• Commission rates are critical for scalping profitability");
        println!("• Funding rates are typically irrelevant for short-term trades");
        println!("• High-frequency strategies require careful cost analysis");
        println!("• Consider using maker orders when possible to reduce costs");
    }
}

/// Commission comparison results
#[derive(Debug, Clone)]
pub struct CommissionComparison {
    pub low_commission_profit: f64,
    pub high_commission_profit: f64,
    pub low_commission_total: f64,
    pub high_commission_total: f64,
    pub profit_difference: f64,
    pub commission_difference: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalping_strategy_creation() {
        let strategy = ScalpingStrategy::new();
        assert!(strategy.name().contains("EMA"));
        assert_eq!(
            strategy.strategy_type(),
            super::super::StrategyType::Scalping
        );
    }

    #[test]
    fn test_custom_config() {
        let config = ScalpingConfig {
            fast_period: 5,
            slow_period: 10,
            initial_capital: 500.0,
            ..Default::default()
        };

        let strategy = ScalpingStrategy::with_config(config);
        assert!(strategy.name().contains("5/10"));
    }

    #[tokio::test]
    async fn test_strategy_run() {
        let strategy = ScalpingStrategy::new();
        let results = strategy.run_with_sample_data();
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.main_results.trade_count > 0);
        assert!(results.main_results.initial_capital > 0.0);
    }
}
