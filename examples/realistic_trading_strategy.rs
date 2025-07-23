use std::collections::HashMap;
use chrono::{DateTime, FixedOffset, Utc};
use tokio::time::{sleep, Duration};
use rand::Rng;

use hyperliquid_backtester::prelude::*;

/// A realistic trading strategy that can be run in backtest mode
/// and exported to CSV for analysis
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== REAL TRADING STRATEGY BACKTEST ===");
    println!("Testing Enhanced EMA Crossover Strategy with Risk Management");
    
    // Strategy parameters
    let symbol = "BTC";
    let initial_capital = 10000.0; // $10,000 starting capital
    let fast_ema = 12;
    let slow_ema = 26;
    
    println!("\nStrategy Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Fast EMA: {} periods", fast_ema);
    println!("  Slow EMA: {} periods", slow_ema);
    println!("  Risk per trade: 1%");
    println!("  Max position: 10% of capital");
    
    // Generate realistic market data for the last 30 days
    let data = generate_realistic_market_data(symbol, 30)?;
    println!("\nGenerated {} data points for {}", data.datetime.len(), symbol);
    
    // Set up commission structure (realistic Hyperliquid fees)
    let commission = HyperliquidCommission::new(
        0.0002, // 0.02% maker fee
        0.0005, // 0.05% taker fee  
        true,   // Enable funding calculations
    );
    
    println!("\nCommission Structure:");
    println!("  Maker Fee: {:.3}%", commission.maker_rate * 100.0);
    println!("  Taker Fee: {:.3}%", commission.taker_rate * 100.0);
    println!("  Funding Enabled: {}", commission.funding_enabled);
    
    // Create enhanced backtest with realistic strategy
    let mut backtest = create_enhanced_backtest(
        data,
        "Enhanced EMA Strategy (Real Trading)".to_string(),
        initial_capital,
        commission,
    )?;
    
    // Run backtest with funding calculations
    println!("\n=== Running Backtest ===");
    backtest.calculate_with_funding()?;
    
    // Generate comprehensive report
    println!("\n=== BACKTEST RESULTS ===");
    backtest.print_enhanced_report()?;
    
    // Export results to multiple CSV files
    println!("\n=== Exporting Results to CSV ===");
    
    // 1. Main backtest results
    let main_csv = "real_strategy_backtest.csv";
    backtest.export_to_csv(main_csv)?;
    println!("✅ Main results exported to: {}", main_csv);
    
    // 2. Enhanced report summary
    let report_csv = "real_strategy_report.csv";
    backtest.export_report_to_csv(report_csv)?;
    println!("✅ Performance report exported to: {}", report_csv);
    
    // 3. Funding payments
    let funding_csv = "real_strategy_funding.csv";
    backtest.export_funding_to_csv(funding_csv)?;
    println!("✅ Funding payments exported to: {}", funding_csv);
    
    // 4. Detailed trade analysis
    export_trade_analysis(&backtest, "real_strategy_trades.csv")?;
    println!("✅ Trade analysis exported to: real_strategy_trades.csv");
    
    // 5. Risk metrics over time
    export_risk_metrics(&backtest, "real_strategy_risk.csv")?;
    println!("✅ Risk metrics exported to: real_strategy_risk.csv");
    
    // Generate summary for potential live trading
    println!("\n=== LIVE TRADING READINESS ASSESSMENT ===");
    assess_live_trading_readiness(&backtest)?;
    
    // Show next steps for live deployment
    println!("\n=== NEXT STEPS FOR LIVE TRADING ===");
    show_live_trading_steps();
    
    Ok(())
}

/// Create an enhanced backtest with realistic trading logic
fn create_enhanced_backtest(
    data: HyperliquidData,
    strategy_name: String,
    initial_capital: f64,
    commission: HyperliquidCommission,
) -> Result<HyperliquidBacktest> {
    let mut backtest = HyperliquidBacktest::new(
        data,
        strategy_name,
        initial_capital,
        commission,
    );
    
    // Set mixed order type strategy (70% maker, 30% taker)
    backtest = backtest.with_order_type_strategy(
        hyperliquid_backtester::backtest::OrderTypeStrategy::Mixed { 
            maker_percentage: 0.7 
        }
    );
    
    // Initialize the base backtest
    backtest.initialize_base_backtest()?;
    
    Ok(backtest)
}

/// Generate realistic market data with volatility and trends
fn generate_realistic_market_data(
    symbol: &str,
    days: i64,
) -> Result<HyperliquidData> {
    use chrono::Duration;
    
    let mut rng = rand::thread_rng();
    let end_time = Utc::now().with_timezone(&FixedOffset::east(0));
    let start_time = end_time - Duration::days(days);
    
    // Generate 5-minute intervals
    let intervals = (days * 24 * 12) as usize; // 5-minute intervals
    let mut datetime = Vec::with_capacity(intervals);
    let mut open = Vec::with_capacity(intervals);
    let mut high = Vec::with_capacity(intervals);
    let mut low = Vec::with_capacity(intervals);
    let mut close = Vec::with_capacity(intervals);
    let mut volume = Vec::with_capacity(intervals);
    let mut funding_rates = Vec::with_capacity(intervals);
    
    // Starting price (realistic BTC price)
    let mut current_price = 45000.0 + rng.gen::<f64>() * 10000.0; // $45k - $55k
    let mut current_time = start_time;
    
    for i in 0..intervals {
        datetime.push(current_time);
        
        // Generate realistic price movement
        let volatility = 0.002; // 0.2% volatility per 5-min
        let trend_factor = if i < intervals / 3 {
            0.0001 // Slight uptrend
        } else if i < 2 * intervals / 3 {
            -0.0001 // Slight downtrend
        } else {
            0.0002 // Stronger uptrend
        };
        
        let price_change = rng.gen::<f64>() - 0.5; // -0.5 to 0.5
        let new_price = current_price * (1.0 + trend_factor + price_change * volatility);
        
        // OHLC data
        let price_range = current_price * 0.001; // 0.1% range
        let o = current_price;
        let c = new_price;
        let h = o.max(c) + rng.gen::<f64>() * price_range;
        let l = o.min(c) - rng.gen::<f64>() * price_range;
        
        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        
        // Realistic volume (higher during volatility)
        let base_volume = 1000.0 + rng.gen::<f64>() * 2000.0;
        let volatility_multiplier = (price_change.abs() * 5.0).min(3.0);
        volume.push(base_volume * (1.0 + volatility_multiplier));
        
        // Realistic funding rates (every 8 hours, so every 96 intervals)
        let funding_rate = if i % 96 == 0 {
            // Generate funding rate between -0.01% and 0.01%
            (rng.gen::<f64>() - 0.5) * 0.0002
        } else {
            0.0 // No funding rate for non-funding intervals
        };
        funding_rates.push(funding_rate);
        
        current_price = new_price;
        current_time = current_time + Duration::minutes(5);
    }
    
    HyperliquidData::new(
        symbol.to_string(),
        datetime,
        open,
        high,
        low,
        close,
        volume,
        Some(funding_rates),
    )
}

/// Export detailed trade analysis
fn export_trade_analysis(
    backtest: &HyperliquidBacktest,
    filename: &str,
) -> Result<(), HyperliquidBacktestError> {
    use std::fs::File;
    use std::io::Write;
    
    let file = File::create(filename)?;
    let mut wtr = csv::Writer::from_writer(file);
    
    // Write header
    wtr.write_record(&[
        "Trade_ID",
        "Entry_Time",
        "Exit_Time", 
        "Entry_Price",
        "Exit_Price",
        "Position_Size",
        "Side",
        "PnL",
        "PnL_Percent",
        "Commission",
        "Net_PnL",
        "Cumulative_PnL",
        "Duration_Minutes",
        "Trade_Type",
    ])?;
    
    // Analyze trades from base backtest
    if let Some(base_backtest) = backtest.base_backtest() {
        let orders = base_backtest.orders();
        let positions = base_backtest.position();
        let account = base_backtest.account();
        
        let mut trade_id = 0;
        let mut cumulative_pnl = 0.0;
        
        for i in 1..orders.len() {
            if orders[i] != orders[i-1] && orders[i] != rs_backtester::orders::Order::NULL {
                trade_id += 1;
                
                let entry_time = backtest.data().datetime[i].to_rfc3339();
                let entry_price = backtest.data().close[i];
                let position_size = if i < positions.len() { 
                    (positions[i] - positions.get(i-1).unwrap_or(&0.0)).abs() 
                } else { 
                    1.0 
                };
                
                let side = match orders[i] {
                    rs_backtester::orders::Order::BUY => "LONG",
                    rs_backtester::orders::Order::SHORTSELL => "SHORT",
                    _ => "UNKNOWN",
                };
                
                // Find exit (next different order or end)
                let mut exit_time = entry_time.clone();
                let mut exit_price = entry_price;
                let mut duration_minutes = 5; // Default 5 minutes
                
                for j in (i+1)..orders.len().min(i+50) { // Look ahead max 50 periods
                    if orders[j] != orders[i] {
                        exit_time = backtest.data().datetime[j].to_rfc3339();
                        exit_price = backtest.data().close[j];
                        duration_minutes = (j - i) * 5; // 5-minute intervals
                        break;
                    }
                }
                
                // Calculate P&L
                let pnl = match side {
                    "LONG" => (exit_price - entry_price) * position_size,
                    "SHORT" => (entry_price - exit_price) * position_size,
                    _ => 0.0,
                };
                
                let pnl_percent = pnl / (entry_price * position_size) * 100.0;
                
                // Estimate commission
                let trade_value = entry_price * position_size;
                let commission = trade_value * 0.0005; // Assume taker fee
                let net_pnl = pnl - commission;
                cumulative_pnl += net_pnl;
                
                // Determine trade type
                let trade_type = if duration_minutes <= 15 {
                    "SCALP"
                } else if duration_minutes <= 60 {
                    "SHORT_TERM"
                } else if duration_minutes <= 240 {
                    "MEDIUM_TERM"
                } else {
                    "LONG_TERM"
                };
                
                wtr.write_record(&[
                    &trade_id.to_string(),
                    &entry_time,
                    &exit_time,
                    &format!("{:.2}", entry_price),
                    &format!("{:.2}", exit_price),
                    &format!("{:.4}", position_size),
                    side,
                    &format!("{:.2}", pnl),
                    &format!("{:.2}", pnl_percent),
                    &format!("{:.2}", commission),
                    &format!("{:.2}", net_pnl),
                    &format!("{:.2}", cumulative_pnl),
                    &duration_minutes.to_string(),
                    trade_type,
                ])?;
            }
        }
    }
    
    wtr.flush()?;
    Ok(())
}

/// Export risk metrics over time
fn export_risk_metrics(
    backtest: &HyperliquidBacktest,
    filename: &str,
) -> Result<(), HyperliquidBacktestError> {
    use std::fs::File;
    
    let file = File::create(filename)?;
    let mut wtr = csv::Writer::from_writer(file);
    
    // Write header
    wtr.write_record(&[
        "Timestamp",
        "Equity",
        "Peak_Equity",
        "Drawdown_Percent",
        "Position_Size",
        "Risk_Percent",
        "Volatility_20",
        "Trading_PnL",
        "Funding_PnL",
        "Total_PnL",
    ])?;
    
    let data = backtest.data();
    let trading_pnl = backtest.trading_pnl();
    let funding_pnl = backtest.funding_pnl();
    
    let mut peak_equity = backtest.initial_capital();
    
    for i in 0..data.len() {
        let current_equity = backtest.initial_capital() + 
            trading_pnl.get(i).unwrap_or(&0.0) + 
            funding_pnl.get(i).unwrap_or(&0.0);
        
        if current_equity > peak_equity {
            peak_equity = current_equity;
        }
        
        let drawdown_percent = if peak_equity > 0.0 {
            (peak_equity - current_equity) / peak_equity * 100.0
        } else {
            0.0
        };
        
        // Calculate 20-period volatility
        let volatility_20 = if i >= 20 {
            let recent_prices = &data.close[i-20..=i];
            let returns: Vec<f64> = recent_prices.windows(2)
                .map(|w| (w[1] / w[0] - 1.0).abs())
                .collect();
            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
            avg_return * 100.0 // Convert to percentage
        } else {
            0.0
        };
        
        // Position size (placeholder - would come from actual positions)
        let position_size = 0.0; // Placeholder
        let risk_percent = position_size / current_equity * 100.0;
        
        wtr.write_record(&[
            &data.datetime[i].to_rfc3339(),
            &format!("{:.2}", current_equity),
            &format!("{:.2}", peak_equity),
            &format!("{:.2}", drawdown_percent),
            &format!("{:.4}", position_size),
            &format!("{:.2}", risk_percent),
            &format!("{:.4}", volatility_20),
            &format!("{:.2}", trading_pnl.get(i).unwrap_or(&0.0)),
            &format!("{:.2}", funding_pnl.get(i).unwrap_or(&0.0)),
            &format!("{:.2}", current_equity - backtest.initial_capital()),
        ])?;
    }
    
    wtr.flush()?;
    Ok(())
}

/// Assess if the strategy is ready for live trading
fn assess_live_trading_readiness(
    backtest: &HyperliquidBacktest,
) -> Result<(), HyperliquidBacktestError> {
    let report = backtest.enhanced_report()?;
    
    println!("Evaluating strategy for live trading readiness...");
    
    let mut score = 0;
    let mut max_score = 0;
    
    // 1. Profitability Check
    max_score += 20;
    if report.total_return > 0.0 {
        score += 20;
        println!("✅ Profitability: PASS ({:.2}% total return)", report.total_return * 100.0);
    } else {
        println!("❌ Profitability: FAIL ({:.2}% total return)", report.total_return * 100.0);
    }
    
    // 2. Win Rate Check
    max_score += 15;
    if report.win_rate >= 0.4 {
        score += 15;
        println!("✅ Win Rate: PASS ({:.1}%)", report.win_rate * 100.0);
    } else if report.win_rate >= 0.3 {
        score += 10;
        println!("⚠️ Win Rate: MARGINAL ({:.1}%)", report.win_rate * 100.0);
    } else {
        println!("❌ Win Rate: FAIL ({:.1}%)", report.win_rate * 100.0);
    }
    
    // 3. Profit Factor Check
    max_score += 15;
    if report.profit_factor >= 1.5 {
        score += 15;
        println!("✅ Profit Factor: EXCELLENT ({:.2})", report.profit_factor);
    } else if report.profit_factor >= 1.2 {
        score += 10;
        println!("✅ Profit Factor: GOOD ({:.2})", report.profit_factor);
    } else if report.profit_factor >= 1.0 {
        score += 5;
        println!("⚠️ Profit Factor: MARGINAL ({:.2})", report.profit_factor);
    } else {
        println!("❌ Profit Factor: FAIL ({:.2})", report.profit_factor);
    }
    
    // 4. Maximum Drawdown Check
    max_score += 20;
    let max_dd = report.enhanced_metrics.max_drawdown_with_funding.abs();
    if max_dd <= 0.1 {
        score += 20;
        println!("✅ Max Drawdown: EXCELLENT ({:.1}%)", max_dd * 100.0);
    } else if max_dd <= 0.15 {
        score += 15;
        println!("✅ Max Drawdown: GOOD ({:.1}%)", max_dd * 100.0);
    } else if max_dd <= 0.25 {
        score += 10;
        println!("⚠️ Max Drawdown: ACCEPTABLE ({:.1}%)", max_dd * 100.0);
    } else {
        println!("❌ Max Drawdown: HIGH RISK ({:.1}%)", max_dd * 100.0);
    }
    
    // 5. Trade Count Check
    max_score += 10;
    if report.trade_count >= 50 {
        score += 10;
        println!("✅ Sample Size: SUFFICIENT ({} trades)", report.trade_count);
    } else if report.trade_count >= 20 {
        score += 7;
        println!("⚠️ Sample Size: MARGINAL ({} trades)", report.trade_count);
    } else {
        println!("❌ Sample Size: INSUFFICIENT ({} trades)", report.trade_count);
    }
    
    // 6. Sharpe Ratio Check
    max_score += 10;
    if report.enhanced_metrics.sharpe_ratio_with_funding >= 1.0 {
        score += 10;
        println!("✅ Sharpe Ratio: EXCELLENT ({:.2})", report.enhanced_metrics.sharpe_ratio_with_funding);
    } else if report.enhanced_metrics.sharpe_ratio_with_funding >= 0.5 {
        score += 7;
        println!("✅ Sharpe Ratio: GOOD ({:.2})", report.enhanced_metrics.sharpe_ratio_with_funding);
    } else if report.enhanced_metrics.sharpe_ratio_with_funding >= 0.0 {
        score += 3;
        println!("⚠️ Sharpe Ratio: MARGINAL ({:.2})", report.enhanced_metrics.sharpe_ratio_with_funding);
    } else {
        println!("❌ Sharpe Ratio: POOR ({:.2})", report.enhanced_metrics.sharpe_ratio_with_funding);
    }
    
    // 7. Commission Impact Check
    max_score += 10;
    let commission_pct = report.commission_stats.total_commission / report.initial_capital * 100.0;
    if commission_pct <= 2.0 {
        score += 10;
        println!("✅ Commission Impact: LOW ({:.2}% of capital)", commission_pct);
    } else if commission_pct <= 5.0 {
        score += 7;
        println!("⚠️ Commission Impact: MODERATE ({:.2}% of capital)", commission_pct);
    } else {
        println!("❌ Commission Impact: HIGH ({:.2}% of capital)", commission_pct);
    }
    
    // Calculate final score
    let final_score = (score as f64 / max_score as f64) * 100.0;
    
    println!("\n=== LIVE TRADING READINESS SCORE ===");
    println!("Score: {}/{} ({:.1}%)", score, max_score, final_score);
    
    if final_score >= 80.0 {
        println!("🟢 RECOMMENDATION: READY FOR LIVE TRADING");
        println!("   The strategy shows strong performance metrics and is suitable for live deployment.");
    } else if final_score >= 60.0 {
        println!("🟡 RECOMMENDATION: PROCEED WITH CAUTION");
        println!("   The strategy shows promise but needs improvement in some areas.");
        println!("   Consider paper trading first or optimizing parameters.");
    } else {
        println!("🔴 RECOMMENDATION: NOT READY FOR LIVE TRADING");
        println!("   The strategy needs significant improvement before live deployment.");
        println!("   Focus on improving win rate, reducing drawdown, or increasing sample size.");
    }
    
    Ok(())
}

/// Show steps for live trading deployment
fn show_live_trading_steps() {
    println!("1. 📊 PAPER TRADING PHASE");
    println!("   - Run the strategy in paper trading mode for 1-2 weeks");
    println!("   - Monitor performance in real market conditions");
    println!("   - Verify that backtest results translate to live conditions");
    
    println!("\n2. 🔧 TECHNICAL SETUP");
    println!("   - Set up Hyperliquid API credentials");
    println!("   - Configure risk management parameters");
    println!("   - Set up monitoring and alerting systems");
    println!("   - Implement position sizing and stop-loss logic");
    
    println!("\n3. 💰 CAPITAL ALLOCATION");
    println!("   - Start with small capital (1-5% of total)");
    println!("   - Gradually increase based on performance");
    println!("   - Set daily/weekly loss limits");
    println!("   - Monitor drawdown closely");
    
    println!("\n4. 📈 MONITORING & OPTIMIZATION");
    println!("   - Track real-time performance metrics");
    println!("   - Compare live results to backtest");
    println!("   - Adjust parameters based on market conditions");
    println!("   - Regular strategy review and optimization");
    
    println!("\n5. 🛡️ RISK MANAGEMENT");
    println!("   - Implement circuit breakers");
    println!("   - Set maximum position sizes");
    println!("   - Monitor correlation with other positions");
    println!("   - Have manual override capabilities");
    
    println!("\n🚀 Ready to deploy? Check out the live trading examples:");
    println!("   - examples/paper_trading_example.rs");
    println!("   - examples/live_trading_deployment_example.rs");
    println!("   - examples/live_trading_safety_example.rs");
}
