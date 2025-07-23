use chrono::{DateTime, Duration, FixedOffset, Utc};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration as TokioDuration};
use tracing::{error, info, trace, warn};

use hyperliquid_backtester::indicators::{bollinger_bands, ema, macd, rsi, sma};
use hyperliquid_backtester::live_trading::{
    AlertLevel, AlertMessage, LiveTradingEngine, LiveTradingError, RetryPolicy,
    SafetyCircuitBreakerConfig,
};
use hyperliquid_backtester::prelude::*;
use hyperliquid_backtester::real_time_monitoring::{
    MonitoringConfig, PerformanceAlert, RealTimeMonitor, RiskAlert,
};
use hyperliquid_backtester::unified_data::{
    FundingPayment, MarketData, OrderRequest, OrderResult, OrderSide, OrderStatus, OrderType,
    Position, Signal, SignalDirection, TimeInForce, TradingStrategy,
};

/// Enhanced EMA Crossover Strategy with Bollinger Bands and RSI filter
/// This is a production-ready strategy that can be used for real trading
pub struct EnhancedEMAStrategy {
    pub name: String,
    pub symbol: String,

    // Strategy parameters
    pub fast_ema: usize,
    pub slow_ema: usize,
    pub bb_period: usize,
    pub bb_deviation: f64,
    pub rsi_period: usize,
    pub rsi_overbought: f64,
    pub rsi_oversold: f64,

    // Position management
    pub base_position_size: f64,
    pub max_position_size: f64,
    pub risk_per_trade: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,

    // State management
    pub current_position: f64,
    pub entry_price: Option<f64>,
    pub last_signal_time: Option<DateTime<FixedOffset>>,
    pub price_history: Vec<f64>,
    pub volume_history: Vec<f64>,
    pub funding_rates: Vec<f64>,

    // Risk management
    pub daily_loss_limit: f64,
    pub daily_pnl: f64,
    pub max_trades_per_day: usize,
    pub trades_today: usize,
    pub last_reset_date: DateTime<FixedOffset>,

    // Performance tracking
    pub total_trades: usize,
    pub winning_trades: usize,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub peak_equity: f64,
}

impl EnhancedEMAStrategy {
    pub fn new(symbol: String, initial_capital: f64) -> Self {
        Self {
            name: "Enhanced EMA Crossover Strategy".to_string(),
            symbol,

            // Default parameters (can be optimized)
            fast_ema: 12,
            slow_ema: 26,
            bb_period: 20,
            bb_deviation: 2.0,
            rsi_period: 14,
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,

            // Position sizing (2% risk per trade)
            base_position_size: initial_capital * 0.02,
            max_position_size: initial_capital * 0.1,
            risk_per_trade: 0.02,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,

            // State
            current_position: 0.0,
            entry_price: None,
            last_signal_time: None,
            price_history: Vec::with_capacity(200),
            volume_history: Vec::with_capacity(200),
            funding_rates: Vec::with_capacity(100),

            // Risk management
            daily_loss_limit: initial_capital * 0.05, // 5% daily loss limit
            daily_pnl: 0.0,
            max_trades_per_day: 10,
            trades_today: 0,
            last_reset_date: Utc::now().with_timezone(&FixedOffset::east(0)),

            // Performance tracking
            total_trades: 0,
            winning_trades: 0,
            total_pnl: 0.0,
            max_drawdown: 0.0,
            peak_equity: initial_capital,
        }
    }

    /// Update price history and calculate indicators
    pub fn update_market_data(&mut self, market_data: &MarketData) {
        // Add new price to history
        self.price_history.push(market_data.close);
        self.volume_history.push(market_data.volume);

        // Keep only last 200 periods
        if self.price_history.len() > 200 {
            self.price_history.remove(0);
            self.volume_history.remove(0);
        }

        // Reset daily counters if new day
        let now = Utc::now().with_timezone(&FixedOffset::east(0));
        if now.date() != self.last_reset_date.date() {
            self.daily_pnl = 0.0;
            self.trades_today = 0;
            self.last_reset_date = now;
        }
    }

    /// Generate trading signal based on multiple indicators
    pub fn generate_signal(
        &self,
        market_data: &MarketData,
        funding_rate: Option<f64>,
    ) -> Option<Signal> {
        // Need enough data for indicators
        if self.price_history.len() < self.slow_ema.max(self.bb_period).max(self.rsi_period) {
            return None;
        }

        // Check risk limits
        if !self.check_risk_limits() {
            trace!("Risk limits exceeded, no signal generated");
            return None;
        }

        // Calculate indicators
        let fast_ema_values = ema(&self.price_history, self.fast_ema);
        let slow_ema_values = ema(&self.price_history, self.slow_ema);
        let bb_bands = bollinger_bands(&self.price_history, self.bb_period, self.bb_deviation);
        let rsi_values = rsi(&self.price_history, self.rsi_period);

        // Get current indicator values
        let fast_ema_current = *fast_ema_values.last().unwrap();
        let slow_ema_current = *slow_ema_values.last().unwrap();
        let fast_ema_prev = fast_ema_values[fast_ema_values.len() - 2];
        let slow_ema_prev = slow_ema_values[slow_ema_values.len() - 2];

        let bb_upper = bb_bands.upper.last().unwrap();
        let bb_lower = bb_bands.lower.last().unwrap();
        let bb_middle = bb_bands.middle.last().unwrap();

        let rsi_current = *rsi_values.last().unwrap();
        let current_price = market_data.close;

        // Determine trend and momentum
        let ema_bullish_cross =
            fast_ema_prev <= slow_ema_prev && fast_ema_current > slow_ema_current;
        let ema_bearish_cross =
            fast_ema_prev >= slow_ema_prev && fast_ema_current < slow_ema_current;

        let price_above_bb_middle = current_price > *bb_middle;
        let price_below_bb_middle = current_price < *bb_middle;

        let rsi_not_overbought = rsi_current < self.rsi_overbought;
        let rsi_not_oversold = rsi_current > self.rsi_oversold;

        // Volume confirmation
        let avg_volume = self.volume_history.iter().take(20).sum::<f64>() / 20.0;
        let volume_confirmation = market_data.volume > avg_volume * 1.2;

        // Funding rate bias (prefer direction that earns funding)
        let funding_bias = if let Some(rate) = funding_rate {
            if rate > 0.0001 {
                Some(SignalDirection::Short) // High positive funding, prefer short
            } else if rate < -0.0001 {
                Some(SignalDirection::Long) // High negative funding, prefer long
            } else {
                None
            }
        } else {
            None
        };

        // Generate buy signal
        if ema_bullish_cross
            && price_above_bb_middle
            && rsi_not_overbought
            && volume_confirmation
            && self.current_position <= 0.0
        {
            // Check funding bias
            if let Some(SignalDirection::Short) = funding_bias {
                trace!("Buy signal suppressed due to funding bias");
                return None;
            }

            let position_size = self.calculate_position_size(current_price, true);

            return Some(Signal {
                timestamp: market_data.timestamp,
                symbol: self.symbol.clone(),
                direction: SignalDirection::Long,
                strength: self.calculate_signal_strength(&market_data, true),
                price: current_price,
                stop_loss: Some(current_price * (1.0 - self.stop_loss_pct)),
                take_profit: Some(current_price * (1.0 + self.take_profit_pct)),
                position_size: Some(position_size),
                reasoning: format!(
                    "EMA bullish cross ({:.2} > {:.2}), price above BB middle, RSI: {:.1}, strong volume",
                    fast_ema_current, slow_ema_current, rsi_current
                ),
                metadata: HashMap::new(),
            });
        }

        // Generate sell signal
        if ema_bearish_cross
            && price_below_bb_middle
            && rsi_not_oversold
            && volume_confirmation
            && self.current_position >= 0.0
        {
            // Check funding bias
            if let Some(SignalDirection::Long) = funding_bias {
                trace!("Sell signal suppressed due to funding bias");
                return None;
            }

            let position_size = self.calculate_position_size(current_price, false);

            return Some(Signal {
                timestamp: market_data.timestamp,
                symbol: self.symbol.clone(),
                direction: SignalDirection::Short,
                strength: self.calculate_signal_strength(&market_data, false),
                price: current_price,
                stop_loss: Some(current_price * (1.0 + self.stop_loss_pct)),
                take_profit: Some(current_price * (1.0 - self.take_profit_pct)),
                position_size: Some(position_size),
                reasoning: format!(
                    "EMA bearish cross ({:.2} < {:.2}), price below BB middle, RSI: {:.1}, strong volume",
                    fast_ema_current, slow_ema_current, rsi_current
                ),
                metadata: HashMap::new(),
            });
        }

        // Check for position exit signals
        if self.current_position != 0.0 {
            if let Some(exit_signal) = self.check_exit_conditions(market_data) {
                return Some(exit_signal);
            }
        }

        None
    }

    /// Calculate position size based on risk management rules
    fn calculate_position_size(&self, current_price: f64, is_long: bool) -> f64 {
        // Base position size
        let mut size = self.base_position_size / current_price;

        // Adjust based on volatility (using ATR proxy)
        if self.price_history.len() >= 14 {
            let recent_prices = &self.price_history[self.price_history.len() - 14..];
            let volatility = recent_prices
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum::<f64>()
                / 13.0;

            let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
            let volatility_pct = volatility / avg_price;

            // Reduce size if high volatility
            if volatility_pct > 0.02 {
                size *= 0.7;
            } else if volatility_pct < 0.005 {
                size *= 1.3;
            }
        }

        // Ensure within limits
        size = size.min(self.max_position_size / current_price);

        // Round to appropriate precision
        (size * 1000.0).round() / 1000.0
    }

    /// Calculate signal strength (0.0 to 1.0)
    fn calculate_signal_strength(&self, market_data: &MarketData, is_bullish: bool) -> f64 {
        let mut strength = 0.5; // Base strength

        if self.price_history.len() < 20 {
            return strength;
        }

        // Volume factor
        let avg_volume = self.volume_history.iter().take(20).sum::<f64>() / 20.0;
        let volume_ratio = market_data.volume / avg_volume;
        strength += (volume_ratio - 1.0).min(0.3).max(-0.2);

        // Trend strength factor
        let trend_strength = if self.price_history.len() >= 5 {
            let recent_prices = &self.price_history[self.price_history.len() - 5..];
            let trend = (recent_prices.last().unwrap() - recent_prices.first().unwrap())
                / recent_prices.first().unwrap();
            if (trend > 0.0) == is_bullish {
                trend.abs().min(0.2)
            } else {
                -trend.abs().max(-0.2)
            }
        } else {
            0.0
        };

        strength += trend_strength;

        // Clamp between 0.1 and 1.0
        strength.max(0.1).min(1.0)
    }

    /// Check if risk limits allow trading
    fn check_risk_limits(&self) -> bool {
        // Daily loss limit
        if self.daily_pnl < -self.daily_loss_limit {
            warn!("Daily loss limit exceeded: {:.2}", self.daily_pnl);
            return false;
        }

        // Daily trade limit
        if self.trades_today >= self.max_trades_per_day {
            warn!("Daily trade limit exceeded: {}", self.trades_today);
            return false;
        }

        // Maximum drawdown check
        let current_equity = self.peak_equity + self.total_pnl;
        let drawdown = (self.peak_equity - current_equity) / self.peak_equity;
        if drawdown > 0.15 {
            // 15% max drawdown
            warn!("Maximum drawdown exceeded: {:.2}%", drawdown * 100.0);
            return false;
        }

        true
    }

    /// Check for position exit conditions
    fn check_exit_conditions(&self, market_data: &MarketData) -> Option<Signal> {
        if let Some(entry_price) = self.entry_price {
            let current_price = market_data.close;
            let is_long = self.current_position > 0.0;

            // Calculate current P&L percentage
            let pnl_pct = if is_long {
                (current_price - entry_price) / entry_price
            } else {
                (entry_price - current_price) / entry_price
            };

            // Stop loss check
            if (is_long && pnl_pct <= -self.stop_loss_pct)
                || (!is_long && pnl_pct <= -self.stop_loss_pct)
            {
                return Some(Signal {
                    timestamp: market_data.timestamp,
                    symbol: self.symbol.clone(),
                    direction: if is_long {
                        SignalDirection::Short
                    } else {
                        SignalDirection::Long
                    },
                    strength: 1.0,
                    price: current_price,
                    stop_loss: None,
                    take_profit: None,
                    position_size: Some(self.current_position.abs()),
                    reasoning: format!("Stop loss triggered: P&L {:.2}%", pnl_pct * 100.0),
                    metadata: HashMap::new(),
                });
            }

            // Take profit check
            if (is_long && pnl_pct >= self.take_profit_pct)
                || (!is_long && pnl_pct >= self.take_profit_pct)
            {
                return Some(Signal {
                    timestamp: market_data.timestamp,
                    symbol: self.symbol.clone(),
                    direction: if is_long {
                        SignalDirection::Short
                    } else {
                        SignalDirection::Long
                    },
                    strength: 0.8,
                    price: current_price,
                    stop_loss: None,
                    take_profit: None,
                    position_size: Some(self.current_position.abs()),
                    reasoning: format!("Take profit triggered: P&L {:.2}%", pnl_pct * 100.0),
                    metadata: HashMap::new(),
                });
            }
        }

        None
    }

    /// Update strategy state after trade execution
    pub fn update_position(&mut self, order_result: &OrderResult) {
        match order_result.status {
            OrderStatus::Filled => {
                let previous_position = self.current_position;

                match order_result.side {
                    OrderSide::Buy => {
                        self.current_position += order_result.filled_qty;
                    }
                    OrderSide::Sell => {
                        self.current_position -= order_result.filled_qty;
                    }
                }

                // Update entry price if opening position
                if previous_position == 0.0 && self.current_position != 0.0 {
                    self.entry_price = Some(order_result.avg_fill_price);
                } else if self.current_position == 0.0 {
                    // Position closed
                    if let Some(entry_price) = self.entry_price {
                        let pnl = if previous_position > 0.0 {
                            (order_result.avg_fill_price - entry_price) * order_result.filled_qty
                        } else {
                            (entry_price - order_result.avg_fill_price) * order_result.filled_qty
                        };

                        self.total_pnl += pnl;
                        self.daily_pnl += pnl;
                        self.total_trades += 1;
                        self.trades_today += 1;

                        if pnl > 0.0 {
                            self.winning_trades += 1;
                        }

                        // Update peak equity and drawdown
                        let current_equity = self.peak_equity + self.total_pnl;
                        if current_equity > self.peak_equity {
                            self.peak_equity = current_equity;
                        }

                        info!(
                            "Position closed: P&L: ${:.2}, Total P&L: ${:.2}, Win Rate: {:.1}%",
                            pnl,
                            self.total_pnl,
                            if self.total_trades > 0 {
                                self.winning_trades as f64 / self.total_trades as f64 * 100.0
                            } else {
                                0.0
                            }
                        );
                    }

                    self.entry_price = None;
                }

                info!(
                    "Position updated: {} {} @ ${:.2}, Position: {:.3}",
                    order_result.side,
                    order_result.filled_qty,
                    order_result.avg_fill_price,
                    self.current_position
                );
            }
            OrderStatus::Rejected => {
                warn!("Order rejected: {:?}", order_result);
            }
            OrderStatus::Cancelled => {
                info!("Order cancelled: {:?}", order_result);
            }
            _ => {}
        }
    }

    /// Get strategy performance summary
    pub fn get_performance_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        summary.insert("total_trades".to_string(), self.total_trades as f64);
        summary.insert("winning_trades".to_string(), self.winning_trades as f64);
        summary.insert(
            "win_rate".to_string(),
            if self.total_trades > 0 {
                self.winning_trades as f64 / self.total_trades as f64
            } else {
                0.0
            },
        );
        summary.insert("total_pnl".to_string(), self.total_pnl);
        summary.insert("daily_pnl".to_string(), self.daily_pnl);
        summary.insert("max_drawdown".to_string(), self.max_drawdown);
        summary.insert("current_position".to_string(), self.current_position);

        summary
    }
}

/// Real Trading Engine that combines strategy with live execution
pub struct RealTradingEngine {
    strategy: EnhancedEMAStrategy,
    live_engine: LiveTradingEngine,
    monitor: RealTimeMonitor,
    is_running: Arc<Mutex<bool>>,
    last_data_update: Option<DateTime<FixedOffset>>,
}

impl RealTradingEngine {
    pub async fn new(symbol: String, initial_capital: f64) -> Result<Self, LiveTradingError> {
        let strategy = EnhancedEMAStrategy::new(symbol.clone(), initial_capital);

        // Configure live trading engine with safety features
        let safety_config = SafetyCircuitBreakerConfig {
            max_daily_loss: initial_capital * 0.05,
            max_position_size: initial_capital * 0.1,
            max_orders_per_minute: 10,
            max_orders_per_day: 100,
            enable_position_limits: true,
            enable_loss_limits: true,
            enable_rate_limits: true,
        };

        let live_engine = LiveTradingEngine::new(safety_config).await?;

        // Configure monitoring
        let monitoring_config = MonitoringConfig {
            enable_performance_alerts: true,
            enable_risk_alerts: true,
            alert_thresholds: HashMap::from([
                ("daily_loss_pct".to_string(), 3.0),
                ("drawdown_pct".to_string(), 10.0),
                ("win_rate_min".to_string(), 30.0),
            ]),
        };

        let monitor = RealTimeMonitor::new(monitoring_config)?;

        Ok(Self {
            strategy,
            live_engine,
            monitor,
            is_running: Arc::new(Mutex::new(false)),
            last_data_update: None,
        })
    }

    /// Start the real trading engine
    pub async fn start(&mut self) -> Result<(), LiveTradingError> {
        info!("Starting real trading engine for {}", self.strategy.symbol);

        // Set running flag
        {
            let mut running = self.is_running.lock().unwrap();
            *running = true;
        }

        // Start monitoring
        self.monitor.start().await?;

        // Main trading loop
        while self.is_running.lock().unwrap().clone() {
            if let Err(e) = self.trading_cycle().await {
                error!("Trading cycle error: {:?}", e);

                // Wait before retrying
                sleep(TokioDuration::from_secs(30)).await;
            } else {
                // Normal cycle delay
                sleep(TokioDuration::from_secs(1)).await;
            }
        }

        info!("Real trading engine stopped");
        Ok(())
    }

    /// Stop the trading engine
    pub async fn stop(&mut self) -> Result<(), LiveTradingError> {
        info!("Stopping real trading engine");

        {
            let mut running = self.is_running.lock().unwrap();
            *running = false;
        }

        // Close any open positions
        if self.strategy.current_position != 0.0 {
            info!(
                "Closing open position: {:.3}",
                self.strategy.current_position
            );

            let close_order = OrderRequest {
                symbol: self.strategy.symbol.clone(),
                side: if self.strategy.current_position > 0.0 {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                },
                order_type: OrderType::Market,
                qty: self.strategy.current_position.abs(),
                price: None,
                stop_price: None,
                time_in_force: TimeInForce::IOC,
                reduce_only: true,
            };

            match self.live_engine.place_order(close_order).await {
                Ok(result) => {
                    info!("Position closed: {:?}", result);
                    self.strategy.update_position(&result);
                }
                Err(e) => {
                    error!("Failed to close position: {:?}", e);
                }
            }
        }

        self.monitor.stop().await?;

        Ok(())
    }

    /// Single trading cycle
    async fn trading_cycle(&mut self) -> Result<(), LiveTradingError> {
        // Get latest market data
        let market_data = self
            .live_engine
            .get_market_data(&self.strategy.symbol)
            .await?;

        // Check if data is fresh
        if let Some(last_update) = self.last_data_update {
            if market_data.timestamp <= last_update {
                return Ok(()); // No new data
            }
        }

        self.last_data_update = Some(market_data.timestamp);

        // Update strategy with new data
        self.strategy.update_market_data(&market_data);

        // Get funding rate
        let funding_rate = self
            .live_engine
            .get_funding_rate(&self.strategy.symbol)
            .await
            .ok();

        // Generate trading signal
        if let Some(signal) = self.strategy.generate_signal(&market_data, funding_rate) {
            info!("Trading signal generated: {:?}", signal);

            // Convert signal to order
            let order = self.signal_to_order(&signal)?;

            // Place order
            match self.live_engine.place_order(order).await {
                Ok(result) => {
                    info!("Order placed successfully: {:?}", result);
                    self.strategy.update_position(&result);

                    // Update monitoring
                    self.monitor.update_trade(&result).await?;
                }
                Err(e) => {
                    error!("Failed to place order: {:?}", e);

                    // Send alert
                    let alert = AlertMessage {
                        level: AlertLevel::Warning,
                        title: "Order Placement Failed".to_string(),
                        message: format!(
                            "Failed to place order for {}: {:?}",
                            self.strategy.symbol, e
                        ),
                        timestamp: Utc::now().with_timezone(&FixedOffset::east(0)),
                    };

                    self.monitor.send_alert(alert).await?;
                }
            }
        }

        // Update performance metrics
        let performance = self.strategy.get_performance_summary();
        self.monitor.update_performance(performance).await?;

        // Check for alerts
        self.check_and_send_alerts().await?;

        Ok(())
    }

    /// Convert signal to order request
    fn signal_to_order(&self, signal: &Signal) -> Result<OrderRequest, LiveTradingError> {
        let side = match signal.direction {
            SignalDirection::Long => OrderSide::Buy,
            SignalDirection::Short => OrderSide::Sell,
        };

        let qty = signal.position_size.unwrap_or(0.0);

        if qty <= 0.0 {
            return Err(LiveTradingError::InvalidOrder(
                "Invalid position size".to_string(),
            ));
        }

        Ok(OrderRequest {
            symbol: signal.symbol.clone(),
            side,
            order_type: OrderType::Market, // Use market orders for simplicity
            qty,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            reduce_only: false,
        })
    }

    /// Check performance and send alerts if needed
    async fn check_and_send_alerts(&mut self) -> Result<(), LiveTradingError> {
        let performance = self.strategy.get_performance_summary();

        // Check daily loss
        let daily_pnl = performance.get("daily_pnl").unwrap_or(&0.0);
        if *daily_pnl < -self.strategy.daily_loss_limit * 0.5 {
            let alert = AlertMessage {
                level: AlertLevel::Warning,
                title: "High Daily Loss".to_string(),
                message: format!(
                    "Daily P&L: ${:.2}, approaching limit of ${:.2}",
                    daily_pnl, -self.strategy.daily_loss_limit
                ),
                timestamp: Utc::now().with_timezone(&FixedOffset::east(0)),
            };

            self.monitor.send_alert(alert).await?;
        }

        // Check win rate
        let win_rate = performance.get("win_rate").unwrap_or(&0.0);
        let total_trades = performance.get("total_trades").unwrap_or(&0.0);

        if *total_trades >= 10.0 && *win_rate < 0.3 {
            let alert = AlertMessage {
                level: AlertLevel::Warning,
                title: "Low Win Rate".to_string(),
                message: format!(
                    "Win rate: {:.1}% over {} trades",
                    win_rate * 100.0,
                    total_trades
                ),
                timestamp: Utc::now().with_timezone(&FixedOffset::east(0)),
            };

            self.monitor.send_alert(alert).await?;
        }

        Ok(())
    }

    /// Get current strategy status
    pub fn get_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();

        status.insert("strategy".to_string(), self.strategy.name.clone());
        status.insert("symbol".to_string(), self.strategy.symbol.clone());
        status.insert(
            "position".to_string(),
            format!("{:.3}", self.strategy.current_position),
        );
        status.insert(
            "total_pnl".to_string(),
            format!("${:.2}", self.strategy.total_pnl),
        );
        status.insert(
            "daily_pnl".to_string(),
            format!("${:.2}", self.strategy.daily_pnl),
        );
        status.insert(
            "trades_today".to_string(),
            self.strategy.trades_today.to_string(),
        );
        status.insert(
            "total_trades".to_string(),
            self.strategy.total_trades.to_string(),
        );
        status.insert(
            "win_rate".to_string(),
            format!(
                "{:.1}%",
                if self.strategy.total_trades > 0 {
                    self.strategy.winning_trades as f64 / self.strategy.total_trades as f64 * 100.0
                } else {
                    0.0
                }
            ),
        );

        status
    }
}

/// Example function to run the real trading strategy
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    hyperliquid_backtester::logging::init_logger()?;

    info!("Starting Real Trading Strategy Example");

    // Strategy parameters
    let symbol = "BTC".to_string();
    let initial_capital = 10000.0; // $10,000

    // Create and start real trading engine
    let mut engine = RealTradingEngine::new(symbol, initial_capital).await?;

    // Set up signal handlers for graceful shutdown
    let engine_running = engine.is_running.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for ctrl+c");
        info!("Shutdown signal received");

        let mut running = engine_running.lock().unwrap();
        *running = false;
    });

    // Start trading
    info!("Real trading engine starting...");
    if let Err(e) = engine.start().await {
        error!("Trading engine error: {:?}", e);
    }

    // Clean shutdown
    engine.stop().await?;

    // Print final performance
    let final_status = engine.get_status();
    info!("Final Performance Summary:");
    for (key, value) in final_status {
        info!("  {}: {}", key, value);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_strategy_creation() {
        let strategy = EnhancedEMAStrategy::new("BTC".to_string(), 10000.0);

        assert_eq!(strategy.symbol, "BTC");
        assert_eq!(strategy.current_position, 0.0);
        assert_eq!(strategy.total_trades, 0);
        assert!(strategy.daily_loss_limit > 0.0);
    }

    #[test]
    fn test_risk_limits() {
        let mut strategy = EnhancedEMAStrategy::new("BTC".to_string(), 10000.0);

        // Test daily loss limit
        strategy.daily_pnl = -600.0; // Exceeds 5% limit
        assert!(!strategy.check_risk_limits());

        // Test daily trade limit
        strategy.daily_pnl = 0.0;
        strategy.trades_today = 11; // Exceeds limit of 10
        assert!(!strategy.check_risk_limits());
    }

    #[test]
    fn test_position_size_calculation() {
        let strategy = EnhancedEMAStrategy::new("BTC".to_string(), 10000.0);
        let position_size = strategy.calculate_position_size(50000.0, true);

        assert!(position_size > 0.0);
        assert!(position_size <= strategy.max_position_size / 50000.0);
    }
}
