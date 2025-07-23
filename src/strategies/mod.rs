//! Production-ready trading strategies module
//!
//! This module contains implementations of various trading strategies optimized for production use.
//! All strategies implement the TradingStrategy trait and can be used with the Hyperliquid backtesting framework.

// Base strategy traits and types
pub mod base;

// Common prelude for all strategies
pub mod prelude;

// Strategy implementations
pub mod momentum;
pub mod scalping;
pub mod swing;

// Live trading components for real-time execution
pub mod live_scalping;

// Re-exports for convenience
pub use base::{StrategyComparison, StrategyFactory, StrategyType, TradingStrategy};
pub use live_scalping::{LiveScalpingEngine, LiveScalpingStrategy, MarketDataFeed, ScalpingSignal};
pub use scalping::ScalpingStrategy;
