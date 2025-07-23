//! Base Strategy Traits and Types
//!
//! This module defines the core interfaces and types for all trading strategies.

use crate::strategies::prelude::*;

/// Core trait that all trading strategies must implement
pub trait TradingStrategy {
    /// Returns the name of the strategy
    fn name(&self) -> &str;

    /// Returns a description of the strategy
    fn description(&self) -> &str;

    /// Returns the preferred timeframe for this strategy
    fn timeframe(&self) -> &str;

    /// Returns the type of strategy
    fn strategy_type(&self) -> StrategyType;

    /// Run the strategy with the provided data
    fn run(&self, data: &HyperliquidData) -> Result<HyperliquidBacktest>;
}

/// Type of trading strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyType {
    /// High-frequency scalping strategy
    Scalping,
    /// Medium-term swing trading strategy  
    Swing,
    /// Momentum-based strategy
    Momentum,
    /// Arbitrage strategy
    Arbitrage,
    /// Market making strategy
    MarketMaking,
    /// Mean reversion strategy
    MeanReversion,
    /// Custom strategy type
    Custom,
}

impl std::fmt::Display for StrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StrategyType::Scalping => write!(f, "Scalping"),
            StrategyType::Swing => write!(f, "Swing Trading"),
            StrategyType::Momentum => write!(f, "Momentum"),
            StrategyType::Arbitrage => write!(f, "Arbitrage"),
            StrategyType::MarketMaking => write!(f, "Market Making"),
            StrategyType::MeanReversion => write!(f, "Mean Reversion"),
            StrategyType::Custom => write!(f, "Custom"),
        }
    }
}

/// Risk level of a strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Extreme,
}

/// Base configuration for all strategies
pub trait StrategyConfig {
    /// Returns the initial capital for the strategy
    fn initial_capital(&self) -> f64;

    /// Returns the commission configuration
    fn commission_config(&self) -> CommissionConfig;

    /// Returns the risk level
    fn risk_level(&self) -> RiskLevel;

    /// Validates the configuration
    fn validate(&self) -> Result<()>;
}

/// Commission configuration for strategies
#[derive(Debug, Clone)]
pub struct CommissionConfig {
    pub maker_rate: f64,
    pub taker_rate: f64,
    pub funding_enabled: bool,
}

impl Default for CommissionConfig {
    fn default() -> Self {
        Self {
            maker_rate: 0.0001,     // 0.01%
            taker_rate: 0.0003,     // 0.03%
            funding_enabled: false,
        }
    }
}

/// Strategy execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Backtest mode - historical data
    Backtest,
    /// Paper trading mode - live data, simulated trades
    Paper,
    /// Live trading mode - real money
    Live,
}

/// Strategy performance metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    pub total_return: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub trade_count: usize,
    pub total_commission: f64,
    pub final_equity: f64,
}

impl StrategyMetrics {
    pub fn from_enhanced_report(report: &EnhancedReport) -> Self {
        Self {
            total_return: report.total_return,
            max_drawdown: report.max_drawdown,
            win_rate: report.win_rate,
            profit_factor: report.profit_factor,
            sharpe_ratio: report.sharpe_ratio,
            trade_count: report.trade_count,
            total_commission: report.commission_stats.total_commission,
            final_equity: report.final_equity,
        }
    }
}

/// Strategy factory for creating different types of strategies
pub struct StrategyFactory;

impl StrategyFactory {
    /// Create a scalping strategy with default configuration
    pub fn create_scalping() -> Box<dyn TradingStrategy> {
        Box::new(crate::strategies::scalping::ScalpingStrategy::new())
    }

    /// Create a scalping strategy with custom configuration
    pub fn create_scalping_with_config(config: crate::strategies::scalping::ScalpingConfig) -> Box<dyn TradingStrategy> {
        Box::new(crate::strategies::scalping::ScalpingStrategy::with_config(config))
    }

    /// Create a strategy by type
    pub fn create_by_type(strategy_type: StrategyType) -> Result<Box<dyn TradingStrategy>> {
        match strategy_type {
            StrategyType::Scalping => Ok(Self::create_scalping()),
            _ => Err(HyperliquidError::validation_error(
                format!("Strategy type {:?} not yet implemented", strategy_type)
            )),
        }
    }
}

/// Strategy comparison utilities
pub struct StrategyComparison;

impl StrategyComparison {
    /// Compare multiple strategies on the same dataset
    pub fn compare_strategies(
        strategies: Vec<Box<dyn TradingStrategy>>, 
        data: &HyperliquidData
    ) -> Result<Vec<(String, StrategyMetrics)>> {
        let mut results = Vec::new();
        
        for strategy in strategies {
            let backtest = strategy.run(data)?;
            let report = backtest.enhanced_report()?;
            let metrics = StrategyMetrics::from_enhanced_report(&report);
            results.push((strategy.name().to_string(), metrics));
        }
        
        Ok(results)
    }

    /// Print comparison results
    pub fn print_comparison(results: &[(String, StrategyMetrics)]) {
        println!("📊 Strategy Comparison Results");
        println!("==============================\n");

        // Header
        println!("{:<25} {:>12} {:>12} {:>10} {:>12} {:>10} {:>8}", 
                 "Strategy", "Return %", "Max DD %", "Win Rate", "Sharpe", "Trades", "P.Factor");
        println!("{}", "-".repeat(100));

        // Results
        for (name, metrics) in results {
            println!(
                "{:<25} {:>11.2}% {:>11.2}% {:>9.2}% {:>11.2} {:>9} {:>7.2}",
                name,
                metrics.total_return * 100.0,
                metrics.max_drawdown * 100.0,
                metrics.win_rate * 100.0,
                metrics.sharpe_ratio,
                metrics.trade_count,
                metrics.profit_factor
            );
        }
        println!();

        // Find best performer
        if let Some((best_name, best_metrics)) = results.iter()
            .max_by(|a, b| a.1.total_return.partial_cmp(&b.1.total_return).unwrap()) {
            println!("🏆 Best Performer: {} ({:.2}% return)", best_name, best_metrics.total_return * 100.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_type_display() {
        assert_eq!(StrategyType::Scalping.to_string(), "Scalping");
        assert_eq!(StrategyType::Momentum.to_string(), "Momentum");
    }

    #[test]
    fn test_commission_config_default() {
        let config = CommissionConfig::default();
        assert_eq!(config.maker_rate, 0.0001);
        assert_eq!(config.taker_rate, 0.0003);
        assert!(!config.funding_enabled);
    }

    #[test]
    fn test_strategy_factory() {
        let strategy = StrategyFactory::create_scalping();
        assert_eq!(strategy.strategy_type(), StrategyType::Scalping);
    }
}
