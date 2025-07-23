//! Live Trading Components for Real-time Scalping
//!
//! This module provides abstractions and utilities for implementing the core scalping
//! patterns identified in the Hyperliquid SDK analysis, bridging backtesting strategies
//! with live trading execution.

use crate::strategies::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Real-time market data feed types based on Hyperliquid SDK analysis
#[derive(Debug, Clone)]
pub enum MarketDataFeed {
    /// Best bid/offer - ultra-low latency (ws_bbo.rs)
    BestBidOffer {
        bid: f64,
        ask: f64,
        timestamp: DateTime<FixedOffset>,
    },
    /// Level 2 order book depth (ws_l2_book.rs)
    OrderBook {
        levels: Vec<BookLevel>,
        timestamp: DateTime<FixedOffset>,
    },
    /// Trade tick stream (ws_trades.rs)
    Trade {
        price: f64,
        size: f64,
        side: TradeSide,
        timestamp: DateTime<FixedOffset>,
    },
}

/// Order book level
#[derive(Debug, Clone)]
pub struct BookLevel {
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Live scalping order management
#[derive(Debug, Clone)]
pub struct ScalpingOrder {
    pub id: String,
    pub symbol: String,
    pub side: TradeSide,
    pub price: f64,
    pub size: f64,
    pub order_type: ScalpingOrderType,
    pub status: OrderStatus,
    pub created_at: DateTime<FixedOffset>,
    pub schedule_cancel_ms: Option<u64>, // Dead-man's switch timing
}

/// Scalping-specific order types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalpingOrderType {
    /// Post-and-pull limit order
    PostAndPull,
    /// Market order for inventory management
    InventoryFlip,
    /// IOC (Immediate or Cancel) for quick execution
    ImmediateOrCancel,
}

/// Order status tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Live scalping strategy trait - extends TradingStrategy for real-time execution
pub trait LiveScalpingStrategy: TradingStrategy {
    /// Process real-time market data and generate trading signals
    fn process_market_data(&mut self, feed: &MarketDataFeed) -> Result<Vec<ScalpingSignal>>;

    /// Calculate optimal post price based on current market conditions
    fn calculate_post_price(&self, side: TradeSide, bbo: &MarketDataFeed) -> Result<f64>;

    /// Determine if current position requires inventory management
    fn needs_inventory_flip(&self, current_position: f64, max_position: f64) -> bool;

    /// Calculate dead-man's switch timing for orders
    fn calculate_cancel_delay_ms(&self) -> u64;

    /// Risk check before placing orders
    fn pre_order_risk_check(
        &self,
        order: &ScalpingOrder,
        account_state: &AccountState,
    ) -> Result<()>;
}

/// Scalping trading signal
#[derive(Debug, Clone)]
pub struct ScalpingSignal {
    pub action: ScalpingAction,
    pub symbol: String,
    pub price: Option<f64>,
    pub size: f64,
    pub urgency: SignalUrgency,
    pub reason: String,
}

/// Scalping actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalpingAction {
    /// Post limit order inside spread
    PostInside,
    /// Join best bid/offer
    JoinBestPrice,
    /// Cancel existing orders
    CancelOrders,
    /// Market order to manage inventory
    MarketOrder,
    /// Flatten position immediately
    FlattenPosition,
}

/// Signal urgency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignalUrgency {
    Low,
    Medium,
    High,
    Critical, // Emergency flatten/risk management
}

/// Account state for risk management
#[derive(Debug, Clone)]
pub struct AccountState {
    pub equity: f64,
    pub margin_used: f64,
    pub positions: HashMap<String, f64>,
    pub open_orders: Vec<ScalpingOrder>,
    pub max_position_size: f64,
    pub daily_pnl: f64,
    pub max_daily_loss: f64,
}

/// Live scalping engine configuration
#[derive(Debug, Clone)]
pub struct LiveScalpingConfig {
    /// Maximum position size per symbol
    pub max_position_size: f64,
    /// Default dead-man's switch delay (5000ms minimum per Hyperliquid)
    pub default_cancel_delay_ms: u64,
    /// Risk limits
    pub max_daily_loss: f64,
    /// Minimum spread width to trade
    pub min_spread_bps: f64,
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: u64,
    /// Enable post-and-pull strategy
    pub enable_post_and_pull: bool,
    /// Enable inventory management
    pub enable_inventory_management: bool,
}

impl Default for LiveScalpingConfig {
    fn default() -> Self {
        Self {
            max_position_size: 1000.0,
            default_cancel_delay_ms: 5000, // Hyperliquid minimum
            max_daily_loss: 500.0,
            min_spread_bps: 1.0, // 1 basis point minimum spread
            max_latency_ms: 100,
            enable_post_and_pull: true,
            enable_inventory_management: true,
        }
    }
}

/// Live scalping execution engine
pub struct LiveScalpingEngine {
    config: LiveScalpingConfig,
    account_state: Arc<RwLock<AccountState>>,
    active_orders: Arc<RwLock<HashMap<String, ScalpingOrder>>>,
    market_data_cache: Arc<RwLock<HashMap<String, MarketDataFeed>>>,
    strategy: Box<dyn LiveScalpingStrategy + Send + Sync>,
}

impl LiveScalpingEngine {
    /// Create new live scalping engine
    pub fn new(
        config: LiveScalpingConfig,
        strategy: Box<dyn LiveScalpingStrategy + Send + Sync>,
    ) -> Self {
        Self {
            config,
            account_state: Arc::new(RwLock::new(AccountState {
                equity: 10000.0,
                margin_used: 0.0,
                positions: HashMap::new(),
                open_orders: Vec::new(),
                max_position_size: 1000.0,
                daily_pnl: 0.0,
                max_daily_loss: 500.0,
            })),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            market_data_cache: Arc::new(RwLock::new(HashMap::new())),
            strategy,
        }
    }

    /// Process incoming market data (to be called from ws_bbo, ws_l2_book, etc.)
    pub async fn process_market_data_update(
        &mut self,
        symbol: &str,
        feed: MarketDataFeed,
    ) -> Result<Vec<ScalpingSignal>> {
        // Update market data cache
        {
            let mut cache = self.market_data_cache.write().await;
            cache.insert(symbol.to_string(), feed.clone());
        }

        // Generate signals from strategy
        let signals = self.strategy.process_market_data(&feed)?;

        // Filter signals based on risk and latency checks
        let mut valid_signals = Vec::new();
        let account_state = self.account_state.read().await;

        for signal in signals {
            if self.validate_signal(&signal, &account_state).await? {
                valid_signals.push(signal);
            }
        }

        Ok(valid_signals)
    }

    /// Execute scalping signals (interface to Hyperliquid SDK order placement)
    pub async fn execute_signals(&mut self, signals: Vec<ScalpingSignal>) -> Result<Vec<String>> {
        let mut order_ids = Vec::new();

        for signal in signals {
            match self.execute_single_signal(signal).await {
                Ok(order_id) => order_ids.push(order_id),
                Err(e) => {
                    error!("Failed to execute signal: {}", e);
                    // Continue with other signals
                }
            }
        }

        Ok(order_ids)
    }

    /// Update order status (to be called from ws_orders, ws_user_events)
    pub async fn update_order_status(
        &mut self,
        order_id: &str,
        new_status: OrderStatus,
    ) -> Result<()> {
        let mut orders = self.active_orders.write().await;

        if let Some(order) = orders.get_mut(order_id) {
            order.status = new_status;

            // Update account state based on order status
            if new_status == OrderStatus::Filled {
                self.handle_order_fill(order).await?;
            }
        }

        Ok(())
    }

    /// Risk management and position tracking
    async fn validate_signal(
        &self,
        signal: &ScalpingSignal,
        account_state: &AccountState,
    ) -> Result<bool> {
        // Check daily loss limits
        if account_state.daily_pnl <= -account_state.max_daily_loss {
            warn!("Daily loss limit exceeded, rejecting signal");
            return Ok(false);
        }

        // Check position limits
        let current_position = account_state.positions.get(&signal.symbol).unwrap_or(&0.0);
        if current_position.abs() >= self.config.max_position_size {
            warn!(
                "Position limit exceeded for {}, rejecting signal",
                signal.symbol
            );
            return Ok(false);
        }

        // Check if spread is wide enough
        if let Some(market_data) = self.market_data_cache.read().await.get(&signal.symbol) {
            if let MarketDataFeed::BestBidOffer { bid, ask, .. } = market_data {
                let spread_bps = ((ask - bid) / ((ask + bid) / 2.0)) * 10000.0;
                if spread_bps < self.config.min_spread_bps {
                    debug!(
                        "Spread too narrow ({:.2} bps), rejecting signal",
                        spread_bps
                    );
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Execute individual signal
    async fn execute_single_signal(&mut self, signal: ScalpingSignal) -> Result<String> {
        let order_id = format!("{}_{}", signal.symbol, Utc::now().timestamp_millis());

        // Create scalping order based on signal
        let order = ScalpingOrder {
            id: order_id.clone(),
            symbol: signal.symbol.clone(),
            side: if signal.action == ScalpingAction::PostInside {
                TradeSide::Buy
            } else {
                TradeSide::Sell
            },
            price: signal.price.unwrap_or(0.0),
            size: signal.size,
            order_type: match signal.action {
                ScalpingAction::PostInside | ScalpingAction::JoinBestPrice => {
                    ScalpingOrderType::PostAndPull
                }
                ScalpingAction::MarketOrder | ScalpingAction::FlattenPosition => {
                    ScalpingOrderType::InventoryFlip
                }
                _ => ScalpingOrderType::ImmediateOrCancel,
            },
            status: OrderStatus::Pending,
            created_at: Utc::now().with_timezone(&FixedOffset::east_opt(0).unwrap()),
            schedule_cancel_ms: Some(self.config.default_cancel_delay_ms),
        };

        // Store order
        {
            let mut orders = self.active_orders.write().await;
            orders.insert(order_id.clone(), order);
        }

        info!(
            "Executing scalping signal: {:?} for {}",
            signal.action, signal.symbol
        );

        // Here you would integrate with actual Hyperliquid SDK calls:
        // - exchange.place_order() for PostInside/JoinBestPrice
        // - exchange.order_and_schedule_cancel() for post-and-pull with dead-man's switch
        // - exchange.market_order() for inventory management

        Ok(order_id)
    }

    /// Handle order fill and update positions
    async fn handle_order_fill(&mut self, order: &ScalpingOrder) -> Result<()> {
        let mut account_state = self.account_state.write().await;

        // Update position
        let position_delta = match order.side {
            TradeSide::Buy => order.size,
            TradeSide::Sell => -order.size,
        };

        *account_state
            .positions
            .entry(order.symbol.clone())
            .or_insert(0.0) += position_delta;

        info!(
            "Order filled: {} {} {} at {}",
            order.symbol,
            if order.side == TradeSide::Buy {
                "BUY"
            } else {
                "SELL"
            },
            order.size,
            order.price
        );

        Ok(())
    }

    /// Get current account state (for monitoring/reporting)
    pub async fn get_account_state(&self) -> AccountState {
        self.account_state.read().await.clone()
    }

    /// Emergency flatten all positions
    pub async fn emergency_flatten(&mut self) -> Result<Vec<String>> {
        warn!("Emergency flatten triggered!");

        let account_state = self.account_state.read().await;
        let mut flatten_signals = Vec::new();

        // Create flatten signals for all open positions
        for (symbol, &position) in &account_state.positions {
            if position.abs() > 0.0001 {
                // Ignore tiny positions
                flatten_signals.push(ScalpingSignal {
                    action: ScalpingAction::FlattenPosition,
                    symbol: symbol.clone(),
                    price: None, // Market order
                    size: position.abs(),
                    urgency: SignalUrgency::Critical,
                    reason: "Emergency flatten".to_string(),
                });
            }
        }

        self.execute_signals(flatten_signals).await
    }
}

/// Utility functions for live scalping
pub mod live_utils {
    use super::*;

    /// Calculate optimal post price inside the spread
    pub fn calculate_inside_price(
        bbo: &MarketDataFeed,
        side: TradeSide,
        tick_size: f64,
    ) -> Option<f64> {
        if let MarketDataFeed::BestBidOffer { bid, ask, .. } = bbo {
            match side {
                TradeSide::Buy => {
                    // Post one tick inside the best bid
                    Some((bid + tick_size).min(ask - tick_size))
                }
                TradeSide::Sell => {
                    // Post one tick inside the best ask
                    Some((ask - tick_size).max(bid + tick_size))
                }
            }
        } else {
            None
        }
    }

    /// Calculate spread in basis points
    pub fn calculate_spread_bps(bbo: &MarketDataFeed) -> Option<f64> {
        if let MarketDataFeed::BestBidOffer { bid, ask, .. } = bbo {
            let mid = (bid + ask) / 2.0;
            Some(((ask - bid) / mid) * 10000.0)
        } else {
            None
        }
    }

    /// Check if market conditions favor scalping
    pub fn is_good_scalping_market(bbo: &MarketDataFeed, min_spread_bps: f64) -> bool {
        if let Some(spread_bps) = calculate_spread_bps(bbo) {
            spread_bps >= min_spread_bps && spread_bps <= 50.0 // Not too wide either
        } else {
            false
        }
    }

    /// Calculate optimal size based on available liquidity
    pub fn calculate_optimal_size(book: &[BookLevel], max_size: f64, side: TradeSide) -> f64 {
        let available_liquidity: f64 = book
            .iter()
            .filter(|level| level.side == side)
            .take(3) // Top 3 levels
            .map(|level| level.size)
            .sum();

        (available_liquidity * 0.5).min(max_size) // Use 50% of available liquidity
    }
}

#[cfg(test)]
mod tests {
    use super::live_utils::*;
    use super::*;

    #[test]
    fn test_spread_calculation() {
        let bbo = MarketDataFeed::BestBidOffer {
            bid: 100.0,
            ask: 100.1,
            timestamp: Utc::now().with_timezone(&FixedOffset::east_opt(0).unwrap()),
        };

        let spread_bps = calculate_spread_bps(&bbo).unwrap();
        assert!((spread_bps - 10.0).abs() < 0.1); // 10 bps spread
    }

    #[test]
    fn test_inside_price_calculation() {
        let bbo = MarketDataFeed::BestBidOffer {
            bid: 100.0,
            ask: 100.1,
            timestamp: Utc::now().with_timezone(&FixedOffset::east_opt(0).unwrap()),
        };

        let buy_price = calculate_inside_price(&bbo, TradeSide::Buy, 0.01).unwrap();
        assert_eq!(buy_price, 100.01); // One tick inside bid

        let sell_price = calculate_inside_price(&bbo, TradeSide::Sell, 0.01).unwrap();
        assert_eq!(sell_price, 100.09); // One tick inside ask
    }

    #[test]
    fn test_scalping_market_conditions() {
        let good_bbo = MarketDataFeed::BestBidOffer {
            bid: 100.0,
            ask: 100.1,
            timestamp: Utc::now().with_timezone(&FixedOffset::east_opt(0).unwrap()),
        };

        assert!(is_good_scalping_market(&good_bbo, 5.0)); // 10 bps spread > 5 bps min

        let tight_bbo = MarketDataFeed::BestBidOffer {
            bid: 100.0,
            ask: 100.01,
            timestamp: Utc::now().with_timezone(&FixedOffset::east_opt(0).unwrap()),
        };

        assert!(!is_good_scalping_market(&tight_bbo, 5.0)); // 1 bps spread < 5 bps min
    }
}
