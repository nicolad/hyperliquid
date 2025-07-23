/// Real Trading Strategy using Hyperliquid Rust SDK
/// This example demonstrates how to integrate the official Hyperliquid SDK
/// for live trading with proper risk management and safety features.

use hyperliquid_backtester::prelude::*;
use hyperliquid_rust_sdk::{
    BaseUrl, InfoClient, ExchangeClient, Message, Subscription,
    ClientOrderRequest, ClientOrder, ClientLimit, MarketOrderParams,
};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::H160;
use tokio::sync::mpsc::unbounded_channel;
use chrono::{DateTime, Utc};
use std::str::FromStr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration, interval};
use log::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

/// Configuration for the Hyperliquid SDK trading strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperliquidTradingConfig {
    /// Initial capital for trading
    pub initial_capital: f64,
    /// Maximum position size as a percentage of capital
    pub max_position_percent: f64,
    /// Stop loss percentage
    pub stop_loss_percent: f64,
    /// Take profit percentage
    pub take_profit_percent: f64,
    /// Trading symbol (e.g., "BTC", "ETH")
    pub symbol: String,
    /// Whether to run in paper trading mode
    pub paper_trading: bool,
    /// Daily loss limit in USD
    pub daily_loss_limit: f64,
    /// Maximum number of trades per day
    pub max_trades_per_day: usize,
    /// Fast EMA period for signals
    pub fast_ema_period: usize,
    /// Slow EMA period for signals
    pub slow_ema_period: usize,
    /// Minimum price change for signal validation
    pub min_price_change: f64,
    /// Maximum slippage tolerance
    pub max_slippage: f64,
}

impl Default for HyperliquidTradingConfig {
    fn default() -> Self {
        Self {
            initial_capital: 1000.0,
            max_position_percent: 0.02, // 2% max position size
            stop_loss_percent: 0.01,    // 1% stop loss
            take_profit_percent: 0.02,  // 2% take profit
            symbol: "BTC".to_string(),
            paper_trading: true,        // Always start with paper trading
            daily_loss_limit: 50.0,     // $50 daily loss limit
            max_trades_per_day: 10,
            fast_ema_period: 12,
            slow_ema_period: 26,
            min_price_change: 0.001,    // 0.1% minimum change
            max_slippage: 0.005,        // 0.5% max slippage
        }
    }
}

/// Trading state for the Hyperliquid strategy
#[derive(Debug, Clone)]
pub struct HyperliquidTradingState {
    pub current_position: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub daily_pnl: f64,
    pub trades_today: usize,
    pub last_price: f64,
    pub entry_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub is_trading_enabled: bool,
    pub fast_ema: Option<f64>,
    pub slow_ema: Option<f64>,
    pub last_signal_time: Option<DateTime<Utc>>,
    pub funding_pnl: f64,
}

impl Default for HyperliquidTradingState {
    fn default() -> Self {
        Self {
            current_position: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            daily_pnl: 0.0,
            trades_today: 0,
            last_price: 0.0,
            entry_price: None,
            stop_loss: None,
            take_profit: None,
            is_trading_enabled: true,
            fast_ema: None,
            slow_ema: None,
            last_signal_time: None,
            funding_pnl: 0.0,
        }
    }
}

/// Trade record for CSV export
#[derive(Debug, Clone, Serialize)]
pub struct TradeRecord {
    pub timestamp: String,
    pub action: String,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub position_after: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub reason: String,
    pub order_id: Option<String>,
}

/// Real-time trading strategy using Hyperliquid SDK
pub struct HyperliquidRealTradingStrategy {
    config: HyperliquidTradingConfig,
    info_client: InfoClient,
    exchange_client: Option<ExchangeClient>,
    wallet_address: Option<H160>,
    state: Arc<Mutex<HyperliquidTradingState>>,
    price_history: Arc<Mutex<Vec<f64>>>,
    trade_records: Arc<Mutex<Vec<TradeRecord>>>,
    funding_history: Arc<Mutex<Vec<f64>>>,
}

#[derive(Debug)]
pub enum TradingSignal {
    Buy { size: f64, reason: String },
    Sell { size: f64, reason: String },
    Hold { reason: String },
    ClosePosition { reason: String },
}

impl HyperliquidRealTradingStrategy {
    /// Create a new Hyperliquid real trading strategy
    pub async fn new(
        config: HyperliquidTradingConfig,
        private_key: Option<String>,
        use_testnet: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let base_url = if use_testnet {
            BaseUrl::Testnet
        } else {
            BaseUrl::Mainnet
        };

        // Create info client
        let info_client = InfoClient::new(None, Some(base_url)).await?;
        
        let (exchange_client, wallet_address) = if let Some(key) = private_key {
            if !config.paper_trading {
                let wallet = LocalWallet::from_str(&key)?;
                let address = wallet.address();
                let exchange_client = ExchangeClient::new(
                    None, 
                    wallet, 
                    Some(base_url), 
                    None, 
                    None
                ).await?;
                
                info!("✅ Connected to Hyperliquid with wallet: {:?}", address);
                (Some(exchange_client), Some(address))
            } else {
                let wallet = LocalWallet::from_str(&key)?;
                let address = wallet.address();
                info!("📝 Paper trading mode - Wallet address: {:?}", address);
                (None, Some(address))
            }
        } else {
            info!("📝 No wallet provided - Paper trading only");
            (None, None)
        };

        Ok(Self {
            config,
            info_client,
            exchange_client,
            wallet_address,
            state: Arc::new(Mutex::new(HyperliquidTradingState::default())),
            price_history: Arc::new(Mutex::new(Vec::new())),
            trade_records: Arc::new(Mutex::new(Vec::new())),
            funding_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Start the real-time trading strategy
    pub async fn start_trading(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting Hyperliquid Real Trading Strategy");
        info!("📊 Configuration: {:?}", self.config);
        
        if self.config.paper_trading {
            warn!("📝 PAPER TRADING MODE - No real money at risk");
        } else {
            warn!("💰 LIVE TRADING MODE - Real money at risk!");
            warn!("⚠️  Ensure you understand the risks and have tested thoroughly");
        }

        // Fetch current user state if we have a wallet
        if let Some(address) = self.wallet_address {
            match self.info_client.user_state(address).await {
                Ok(user_state) => {
                    info!("💼 Account Balance: ${:.2}", 
                        user_state.cross_margin_summary.account_value.parse::<f64>().unwrap_or(0.0)
                    );
                    
                    // Check for existing positions
                    for position in &user_state.asset_positions {
                        if position.position.coin == self.config.symbol {
                            let pos_size = position.position.szi.parse::<f64>().unwrap_or(0.0);
                            if pos_size != 0.0 {
                                warn!("⚠️  Existing {} position detected: {:.4}", 
                                    self.config.symbol, pos_size
                                );
                                
                                let mut state = self.state.lock().unwrap();
                                state.current_position = pos_size;
                                
                                // Try to get entry price from recent fills
                                if let Ok(fills) = self.info_client.user_fills(address).await {
                                    for fill in fills.iter().take(10) { // Check last 10 fills
                                        if fill.coin == self.config.symbol {
                                            state.entry_price = Some(fill.px.parse().unwrap_or(0.0));
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => error!("❌ Failed to fetch user state: {}", e),
            }
        }

        // Subscribe to real-time data
        let (price_sender, mut price_receiver) = unbounded_channel();
        let subscription_id = self.info_client
            .subscribe(Subscription::AllMids, price_sender)
            .await?;

        info!("📊 Subscribed to real-time price feeds");

        // Subscribe to candle data for more detailed analysis
        let (candle_sender, mut candle_receiver) = unbounded_channel();
        let candle_subscription_id = self.info_client
            .subscribe(
                Subscription::Candle {
                    coin: self.config.symbol.clone(),
                    interval: "1m".to_string(),
                },
                candle_sender,
            )
            .await?;

        info!("📈 Subscribed to 1-minute candle data for {}", self.config.symbol);

        // Subscribe to user updates if we have a wallet
        let user_updates_subscription_id = if let Some(address) = self.wallet_address {
            let (user_sender, mut user_receiver) = unbounded_channel();
            let sub_id = self.info_client
                .subscribe(
                    Subscription::User { user: address },
                    user_sender,
                )
                .await?;

            // Handle user updates in a separate task
            let state_clone = Arc::clone(&self.state);
            let trade_records_clone = Arc::clone(&self.trade_records);
            tokio::spawn(async move {
                while let Some(Message::User(user_data)) = user_receiver.recv().await {
                    Self::handle_user_update(user_data, &state_clone, &trade_records_clone).await;
                }
            });

            info!("👤 Subscribed to user updates");
            Some(sub_id)
        } else {
            None
        };

        // Risk management timer
        let risk_state = Arc::clone(&self.state);
        let risk_config = self.config.clone();
        tokio::spawn(async move {
            let mut risk_timer = interval(Duration::from_secs(30)); // Check every 30 seconds
            loop {
                risk_timer.tick().await;
                Self::check_risk_limits(&risk_state, &risk_config).await;
            }
        });

        // Performance reporting timer
        let perf_state = Arc::clone(&self.state);
        let perf_records = Arc::clone(&self.trade_records);
        tokio::spawn(async move {
            let mut report_timer = interval(Duration::from_secs(300)); // Report every 5 minutes
            loop {
                report_timer.tick().await;
                Self::print_performance_report(&perf_state, &perf_records).await;
            }
        });

        // Handle candle data for technical analysis
        let candle_state = Arc::clone(&self.state);
        let candle_history = Arc::clone(&self.price_history);
        let candle_config = self.config.clone();
        tokio::spawn(async move {
            while let Some(Message::Candle(candle)) = candle_receiver.recv().await {
                Self::process_candle_data(candle, &candle_state, &candle_history, &candle_config).await;
            }
        });

        // Main trading loop - process price updates
        info!("🔄 Starting main trading loop...");
        while let Some(message) = price_receiver.recv().await {
            match message {
                Message::AllMids(mids_data) => {
                    if let Some(price_str) = mids_data.mids.get(&self.config.symbol) {
                        if let Ok(price) = price_str.parse::<f64>() {
                            // Update current price
                            {
                                let mut state = self.state.lock().unwrap();
                                state.last_price = price;
                                
                                // Update unrealized PnL
                                if let Some(entry_price) = state.entry_price {
                                    if state.current_position != 0.0 {
                                        state.unrealized_pnl = if state.current_position > 0.0 {
                                            (price - entry_price) * state.current_position
                                        } else {
                                            (entry_price - price) * state.current_position.abs()
                                        };
                                    }
                                }
                            }

                            // Generate and execute trading signals
                            let signal = self.generate_trading_signal(price).await;
                            if let Err(e) = self.execute_trading_signal(signal, price).await {
                                error!("❌ Error executing trading signal: {}", e);
                            }
                        }
                    }
                }
                _ => {
                    // Handle other message types if needed
                    debug!("📨 Received other message type");
                }
            }
        }

        // Cleanup subscriptions
        info!("🛑 Stopping trading strategy...");
        self.info_client.unsubscribe(subscription_id).await?;
        self.info_client.unsubscribe(candle_subscription_id).await?;
        if let Some(sub_id) = user_updates_subscription_id {
            self.info_client.unsubscribe(sub_id).await?;
        }

        info!("✅ Trading strategy stopped successfully");
        Ok(())
    }

    /// Generate trading signal based on EMA crossover and risk management
    async fn generate_trading_signal(&self, current_price: f64) -> TradingSignal {
        let state = self.state.lock().unwrap();
        let price_history = self.price_history.lock().unwrap();

        // Check basic conditions
        if !state.is_trading_enabled {
            return TradingSignal::Hold {
                reason: "Trading disabled due to risk limits".to_string(),
            };
        }

        if state.trades_today >= self.config.max_trades_per_day {
            return TradingSignal::Hold {
                reason: "Daily trade limit reached".to_string(),
            };
        }

        if state.daily_pnl <= -self.config.daily_loss_limit {
            return TradingSignal::ClosePosition {
                reason: "Daily loss limit reached".to_string(),
            };
        }

        // Need sufficient price history for EMA calculations
        if price_history.len() < self.config.slow_ema_period {
            return TradingSignal::Hold {
                reason: "Insufficient price history".to_string(),
            };
        }

        // Check for stop loss or take profit
        if let Some(entry_price) = state.entry_price {
            if state.current_position != 0.0 {
                let pnl_percent = if state.current_position > 0.0 {
                    (current_price - entry_price) / entry_price
                } else {
                    (entry_price - current_price) / entry_price
                };

                if pnl_percent <= -self.config.stop_loss_percent {
                    return TradingSignal::ClosePosition {
                        reason: format!("Stop loss triggered: {:.2}%", pnl_percent * 100.0),
                    };
                }

                if pnl_percent >= self.config.take_profit_percent {
                    return TradingSignal::ClosePosition {
                        reason: format!("Take profit triggered: {:.2}%", pnl_percent * 100.0),
                    };
                }
            }
        }

        // Calculate EMAs
        let fast_ema = Self::calculate_ema(&price_history, self.config.fast_ema_period);
        let slow_ema = Self::calculate_ema(&price_history, self.config.slow_ema_period);

        // Update EMAs in state
        drop(state);
        {
            let mut state = self.state.lock().unwrap();
            state.fast_ema = Some(fast_ema);
            state.slow_ema = Some(slow_ema);
        }

        let position_size = self.calculate_position_size(current_price);

        // Generate signals based on EMA crossover
        let state = self.state.lock().unwrap();
        if fast_ema > slow_ema && state.current_position <= 0.0 {
            // Bullish signal
            TradingSignal::Buy {
                size: position_size,
                reason: format!("EMA crossover bullish: Fast({:.2}) > Slow({:.2})", fast_ema, slow_ema),
            }
        } else if fast_ema < slow_ema && state.current_position >= 0.0 {
            // Bearish signal
            TradingSignal::Sell {
                size: position_size,
                reason: format!("EMA crossover bearish: Fast({:.2}) < Slow({:.2})", fast_ema, slow_ema),
            }
        } else {
            TradingSignal::Hold {
                reason: "No clear signal".to_string(),
            }
        }
    }

    /// Execute trading signal
    async fn execute_trading_signal(
        &self,
        signal: TradingSignal,
        current_price: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match signal {
            TradingSignal::Buy { size, reason } => {
                info!("📈 BUY Signal: Size={:.4}, Price=${:.2}, Reason={}", size, current_price, reason);
                self.execute_order(true, size, current_price, reason).await?;
            }
            TradingSignal::Sell { size, reason } => {
                info!("📉 SELL Signal: Size={:.4}, Price=${:.2}, Reason={}", size, current_price, reason);
                self.execute_order(false, size, current_price, reason).await?;
            }
            TradingSignal::ClosePosition { reason } => {
                let position_size = {
                    let state = self.state.lock().unwrap();
                    state.current_position
                };

                if position_size != 0.0 {
                    let close_size = position_size.abs();
                    let is_buy = position_size < 0.0; // Close short = buy, close long = sell
                    
                    info!("🔄 CLOSE Position: Size={:.4}, Price=${:.2}, Reason={}", 
                        close_size, current_price, reason);
                    
                    self.execute_close_order(is_buy, close_size, current_price, reason).await?;
                }
            }
            TradingSignal::Hold { reason: _ } => {
                // No action for hold signals
            }
        }

        Ok(())
    }

    /// Execute a regular trading order
    async fn execute_order(
        &self,
        is_buy: bool,
        size: f64,
        price: f64,
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.paper_trading {
            if let Some(ref exchange_client) = self.exchange_client {
                // Calculate limit price with slippage tolerance
                let limit_price = if is_buy {
                    price * (1.0 + self.config.max_slippage)
                } else {
                    price * (1.0 - self.config.max_slippage)
                };

                let order = ClientOrderRequest {
                    asset: self.config.symbol.clone(),
                    is_buy,
                    reduce_only: false,
                    limit_px: limit_price,
                    sz: size,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(), // Immediate or Cancel
                    }),
                };

                match exchange_client.order(order, None).await {
                    Ok(response) => {
                        info!("✅ Order placed successfully: {:?}", response);
                        self.update_position_after_order(is_buy, size, price, reason, None).await;
                    }
                    Err(e) => {
                        error!("❌ Failed to place order: {}", e);
                        return Err(e.into());
                    }
                }
            }
        } else {
            // Paper trading
            info!("📝 Paper Trade: {} {:.4} {} at ${:.2}", 
                if is_buy { "BUY" } else { "SELL" }, size, self.config.symbol, price);
            self.update_position_after_order(is_buy, size, price, reason, None).await;
        }

        Ok(())
    }

    /// Execute a position closing order
    async fn execute_close_order(
        &self,
        is_buy: bool,
        size: f64,
        price: f64,
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.paper_trading {
            if let Some(ref exchange_client) = self.exchange_client {
                let limit_price = if is_buy {
                    price * (1.0 + self.config.max_slippage)
                } else {
                    price * (1.0 - self.config.max_slippage)
                };

                let order = ClientOrderRequest {
                    asset: self.config.symbol.clone(),
                    is_buy,
                    reduce_only: true, // This is a closing order
                    limit_px: limit_price,
                    sz: size,
                    cloid: None,
                    order_type: ClientOrder::Limit(ClientLimit {
                        tif: "Ioc".to_string(),
                    }),
                };

                match exchange_client.order(order, None).await {
                    Ok(response) => {
                        info!("✅ Close order placed successfully: {:?}", response);
                        self.close_position(price, reason).await;
                    }
                    Err(e) => {
                        error!("❌ Failed to place close order: {}", e);
                        return Err(e.into());
                    }
                }
            }
        } else {
            // Paper trading
            info!("📝 Paper Close: {} {:.4} {} at ${:.2}", 
                if is_buy { "BUY" } else { "SELL" }, size, self.config.symbol, price);
            self.close_position(price, reason).await;
        }

        Ok(())
    }

    /// Update position after order execution
    async fn update_position_after_order(
        &self,
        is_buy: bool,
        size: f64,
        price: f64,
        reason: String,
        order_id: Option<String>,
    ) {
        let mut state = self.state.lock().unwrap();
        
        let size_change = if is_buy { size } else { -size };
        state.current_position += size_change;
        
        // Set entry price if opening a new position
        if state.entry_price.is_none() || 
           (state.current_position > 0.0 && size_change > 0.0) ||
           (state.current_position < 0.0 && size_change < 0.0) {
            state.entry_price = Some(price);
        }
        
        state.trades_today += 1;
        state.last_signal_time = Some(Utc::now());

        // Record the trade
        let mut trade_records = self.trade_records.lock().unwrap();
        trade_records.push(TradeRecord {
            timestamp: Utc::now().to_rfc3339(),
            action: if is_buy { "BUY" } else { "SELL" }.to_string(),
            symbol: self.config.symbol.clone(),
            price,
            size,
            side: if is_buy { "LONG" } else { "SHORT" }.to_string(),
            position_after: state.current_position,
            unrealized_pnl: state.unrealized_pnl,
            realized_pnl: state.realized_pnl,
            reason,
            order_id,
        });

        info!("📊 Position updated: {:.4} {} (Entry: ${:.2})", 
            state.current_position, self.config.symbol, state.entry_price.unwrap_or(0.0));
    }

    /// Close position and calculate realized PnL
    async fn close_position(&self, exit_price: f64, reason: String) {
        let mut state = self.state.lock().unwrap();
        
        if let Some(entry_price) = state.entry_price {
            let realized_pnl = if state.current_position > 0.0 {
                (exit_price - entry_price) * state.current_position
            } else {
                (entry_price - exit_price) * state.current_position.abs()
            };
            
            state.realized_pnl += realized_pnl;
            state.daily_pnl += realized_pnl;
            state.unrealized_pnl = 0.0;
            
            info!("💰 Position closed - Realized PnL: ${:.2}, Total PnL: ${:.2}", 
                realized_pnl, state.realized_pnl);
        }
        
        state.current_position = 0.0;
        state.entry_price = None;
        state.trades_today += 1;

        // Record the close trade
        let mut trade_records = self.trade_records.lock().unwrap();
        trade_records.push(TradeRecord {
            timestamp: Utc::now().to_rfc3339(),
            action: "CLOSE".to_string(),
            symbol: self.config.symbol.clone(),
            price: exit_price,
            size: 0.0,
            side: "FLAT".to_string(),
            position_after: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: state.realized_pnl,
            reason,
            order_id: None,
        });
    }

    /// Calculate position size based on risk management
    fn calculate_position_size(&self, current_price: f64) -> f64 {
        let max_position_value = self.config.initial_capital * self.config.max_position_percent;
        let position_size = max_position_value / current_price;
        
        // Round to 4 decimal places
        (position_size * 10000.0).round() / 10000.0
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        
        if prices.len() < period {
            // Use SMA if not enough data
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for &price in prices.iter().skip(1) {
            ema = (price * multiplier) + (ema * (1.0 - multiplier));
        }

        ema
    }

    /// Process candle data for technical analysis
    async fn process_candle_data(
        candle: hyperliquid_rust_sdk::ws::sub_structs::CandleData,
        state: &Arc<Mutex<HyperliquidTradingState>>,
        price_history: &Arc<Mutex<Vec<f64>>>,
        _config: &HyperliquidTradingConfig,
    ) {
        let close_price = candle.c.parse::<f64>().unwrap_or(0.0);
        
        {
            let mut history = price_history.lock().unwrap();
            history.push(close_price);
            
            // Keep only the last 200 candles for analysis
            if history.len() > 200 {
                history.remove(0);
            }
        }

        debug!("📊 Processed candle: OHLCV({:.2}, {:.2}, {:.2}, {:.2}, {})", 
            candle.o.parse::<f64>().unwrap_or(0.0),
            candle.h.parse::<f64>().unwrap_or(0.0),
            candle.l.parse::<f64>().unwrap_or(0.0),
            close_price,
            candle.v.parse::<f64>().unwrap_or(0.0)
        );
    }

    /// Handle user account updates
    async fn handle_user_update(
        _user_data: hyperliquid_rust_sdk::ws::sub_structs::User,
        _state: &Arc<Mutex<HyperliquidTradingState>>,
        _trade_records: &Arc<Mutex<Vec<TradeRecord>>>,
    ) {
        // Process user updates like fills, funding payments, etc.
        debug!("👤 Received user account update");
    }

    /// Check risk limits and safety controls
    async fn check_risk_limits(
        state: &Arc<Mutex<HyperliquidTradingState>>,
        config: &HyperliquidTradingConfig,
    ) {
        let mut trading_state = state.lock().unwrap();
        
        if trading_state.daily_pnl <= -config.daily_loss_limit {
            trading_state.is_trading_enabled = false;
            warn!("🚨 RISK ALERT: Daily loss limit reached (${:.2}). Trading disabled!", 
                config.daily_loss_limit);
        }

        if trading_state.trades_today >= config.max_trades_per_day {
            warn!("🚨 RISK ALERT: Daily trade limit reached ({} trades)", 
                config.max_trades_per_day);
        }

        // Check for excessive unrealized losses
        let max_unrealized_loss = config.initial_capital * 0.05; // 5% of capital
        if trading_state.unrealized_pnl < -max_unrealized_loss {
            warn!("🚨 RISK ALERT: Large unrealized loss detected: ${:.2}", 
                trading_state.unrealized_pnl);
        }
    }

    /// Print performance report
    async fn print_performance_report(
        state: &Arc<Mutex<HyperliquidTradingState>>,
        trade_records: &Arc<Mutex<Vec<TradeRecord>>>,
    ) {
        let trading_state = state.lock().unwrap();
        let records = trade_records.lock().unwrap();

        info!("📊 === PERFORMANCE REPORT ===");
        info!("💰 Current Price: ${:.2}", trading_state.last_price);
        info!("📍 Current Position: {:.4}", trading_state.current_position);
        info!("📈 Unrealized PnL: ${:.2}", trading_state.unrealized_pnl);
        info!("💵 Realized PnL: ${:.2}", trading_state.realized_pnl);
        info!("📅 Daily PnL: ${:.2}", trading_state.daily_pnl);
        info!("🔢 Trades Today: {}", trading_state.trades_today);
        info!("📋 Total Trade Records: {}", records.len());
        
        if let (Some(fast_ema), Some(slow_ema)) = (trading_state.fast_ema, trading_state.slow_ema) {
            info!("📊 Fast EMA: {:.2}, Slow EMA: {:.2}", fast_ema, slow_ema);
        }
        
        info!("🔄 Trading Enabled: {}", trading_state.is_trading_enabled);
        info!("================================");
    }

    /// Export all trading data to CSV files
    pub async fn export_trading_data(&self, base_filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Export trade records
        let trade_records = self.trade_records.lock().unwrap();
        let trades_filename = format!("{}_trades.csv", base_filename);
        
        let mut wtr = csv::Writer::from_path(&trades_filename)?;
        for record in trade_records.iter() {
            wtr.serialize(record)?;
        }
        wtr.flush()?;
        
        info!("📊 Trade records exported to: {}", trades_filename);

        // Export price history
        let price_history = self.price_history.lock().unwrap();
        let prices_filename = format!("{}_prices.csv", base_filename);
        
        let mut price_wtr = csv::Writer::from_path(&prices_filename)?;
        price_wtr.write_record(&["timestamp", "price"])?;
        
        for (i, &price) in price_history.iter().enumerate() {
            let timestamp = Utc::now() - chrono::Duration::minutes(price_history.len() as i64 - i as i64);
            price_wtr.write_record(&[timestamp.to_rfc3339(), price.to_string()])?;
        }
        price_wtr.flush()?;
        
        info!("📈 Price history exported to: {}", prices_filename);

        // Export current state summary
        let state = self.state.lock().unwrap();
        let summary_filename = format!("{}_summary.csv", base_filename);
        
        let mut summary_wtr = csv::Writer::from_path(&summary_filename)?;
        summary_wtr.write_record(&["metric", "value"])?;
        summary_wtr.write_record(&["current_position", &state.current_position.to_string()])?;
        summary_wtr.write_record(&["unrealized_pnl", &state.unrealized_pnl.to_string()])?;
        summary_wtr.write_record(&["realized_pnl", &state.realized_pnl.to_string()])?;
        summary_wtr.write_record(&["daily_pnl", &state.daily_pnl.to_string()])?;
        summary_wtr.write_record(&["trades_today", &state.trades_today.to_string()])?;
        summary_wtr.write_record(&["last_price", &state.last_price.to_string()])?;
        summary_wtr.flush()?;
        
        info!("📋 Performance summary exported to: {}", summary_filename);

        Ok(())
    }

    /// Get current trading status
    pub fn get_status(&self) -> HyperliquidTradingState {
        self.state.lock().unwrap().clone()
    }

    /// Stop trading gracefully
    pub async fn stop_trading(&self) {
        let mut state = self.state.lock().unwrap();
        state.is_trading_enabled = false;
        info!("🛑 Trading stopped by user request");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    info!("🚀 Hyperliquid SDK Real Trading Strategy");
    info!("⚠️  This is experimental software. Use at your own risk!");
    info!("💡 Always test with paper trading first!");

    // Configuration for conservative trading
    let config = HyperliquidTradingConfig {
        initial_capital: 100.0,      // $100 starting capital
        max_position_percent: 0.01,  // 1% of capital per trade
        stop_loss_percent: 0.005,    // 0.5% stop loss
        take_profit_percent: 0.01,   // 1% take profit
        symbol: "BTC".to_string(),
        paper_trading: true,         // ALWAYS start with paper trading!
        daily_loss_limit: 5.0,       // $5 daily loss limit
        max_trades_per_day: 5,       // Maximum 5 trades per day
        fast_ema_period: 12,
        slow_ema_period: 26,
        min_price_change: 0.001,     // 0.1% minimum change
        max_slippage: 0.005,         // 0.5% max slippage
    };

    // For live trading, uncomment and add your private key:
    // let private_key = Some("your_private_key_here".to_string());
    let private_key = None; // Paper trading mode

    // Create the strategy
    let mut strategy = HyperliquidRealTradingStrategy::new(
        config,
        private_key,
        true, // Use testnet for safety
    ).await?;

    // Set up graceful shutdown
    let strategy_clone = Arc::new(strategy);
    let shutdown_strategy = Arc::clone(&strategy_clone);
    
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        info!("🛑 Shutdown signal received, stopping trading...");
        shutdown_strategy.stop_trading().await;
    });

    // Start trading
    let mut strategy = Arc::try_unwrap(strategy_clone).unwrap();
    
    // Run for a limited time in this example (remove this in production)
    let trading_future = strategy.start_trading();
    let timeout_future = sleep(Duration::from_secs(300)); // Run for 5 minutes

    tokio::select! {
        result = trading_future => {
            if let Err(e) = result {
                error!("❌ Trading strategy error: {}", e);
            }
        }
        _ = timeout_future => {
            info!("⏰ Demo time limit reached, stopping...");
            strategy.stop_trading().await;
        }
    }

    // Export results
    strategy.export_trading_data("hyperliquid_trading_session").await?;

    let final_status = strategy.get_status();
    info!("🏁 Final Trading Status: {:?}", final_status);
    
    info!("✅ Trading session completed successfully");
    
    Ok(())
}
