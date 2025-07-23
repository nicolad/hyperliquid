//! Momentum Trading Strategy Implementation
//!
//! This module will contain momentum-based trading strategies.

use crate::strategies::prelude::*;

/// Placeholder for momentum trading strategy
pub struct MomentumStrategy {
    name: String,
}

impl MomentumStrategy {
    pub fn new() -> Self {
        Self {
            name: "Momentum Strategy (Coming Soon)".to_string(),
        }
    }
}

impl TradingStrategy for MomentumStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Momentum-based trading strategy (implementation coming soon)"
    }

    fn timeframe(&self) -> &str {
        "15m"
    }

    fn strategy_type(&self) -> StrategyType {
        StrategyType::Momentum
    }

    fn run(&self, _data: &HyperliquidData) -> Result<HyperliquidBacktest> {
        Err(HyperliquidError::validation_error(
            "Momentum strategy not yet implemented".to_string()
        ))
    }
}

impl Default for MomentumStrategy {
    fn default() -> Self {
        Self::new()
    }
}
