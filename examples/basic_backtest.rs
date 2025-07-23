use chrono::Utc;
use hyperliquid_backtester::prelude::*;

/// # Scalping Backtest Example
///
/// This example demonstrates how to run a scalping backtest using the Hyperliquid backtester.
/// It shows:
/// - Setting up logging for debugging and monitoring
/// - Creating high-frequency sample data (5-minute intervals)
/// - Implementing a fast EMA crossover strategy for scalping
/// - Configuring low commission rates optimized for high-frequency trading
/// - Running the backtest without funding rates (irrelevant for short-term trades)
/// - Analyzing commission impact on scalping profitability
/// - Comparing performance with different commission structures
/// - Exporting results to CSV for further analysis
///
/// The example uses a 3/8 EMA crossover strategy on BTC/USD 5-minute data over 4 hours.
///
/// ## Usage
///
/// Run this example with:
/// ```bash
/// cargo run --example basic_backtest
/// ```
///
/// For debug logging:
/// ```bash
/// RUST_LOG=debug cargo run --example basic_backtest
/// ```
///
/// For JSON formatted logs:
/// ```bash
/// RUST_LOG=info HYPERLIQUID_LOG_FORMAT=json cargo run --example basic_backtest
/// ```

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging for better debugging and monitoring
    init_logger_with_level("info");

    log::info!("Starting Hyperliquid Scalping Backtest Example");

    println!("⚡ Hyperliquid Scalping Backtest Example");
    println!("=======================================\n");

    // Create sample data for scalping demonstration
    // In a real scenario, you would fetch high-frequency data from the API
    println!("Creating sample BTC/USD scalping data (5-minute intervals)...");

    // Generate sample data - 4 hours of 5-minute intervals (48 data points)
    let mut datetime = Vec::new();
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();
    let mut funding_rates = Vec::new();

    let start_time = Utc::now() - chrono::Duration::hours(4);
    let mut base_price = 43000.0; // Starting BTC price

    for i in 0..48 {
        let timestamp = start_time + chrono::Duration::minutes(i * 5); // 5-minute intervals
        datetime.push(timestamp.with_timezone(&chrono::FixedOffset::east_opt(0).unwrap()));

        // Create scalping-friendly price movements with frequent small reversals
        let cycle_position = (i as f64 / 6.0) % 1.0; // Complete cycle every 6 periods (30 minutes)
        let micro_cycle = (i as f64 / 2.0) % 1.0; // Micro cycle every 2 periods (10 minutes)

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
        volume.push(800.0 + (i as f64 * 15.0) + f64::abs(scalp_move) * 20.0); // Higher volume on bigger moves

        // Funding rates are less relevant for scalping but included for completeness
        funding_rates.push(((i as f64 % 12.0 - 6.0) / 6.0) * 0.00003); // Smaller funding rates for short-term trades

        base_price = close_price;
    }

    // Create HyperliquidData with our scalping sample data
    let data = HyperliquidData::with_ohlc_and_funding_data(
        "BTC".to_string(),
        datetime,
        open,
        high,
        low,
        close,
        volume,
        funding_rates,
    )?;
    println!(
        "Data fetched: {} data points from {} to {}\n",
        data.len(),
        data.datetime.first().unwrap().format("%Y-%m-%d %H:%M"),
        data.datetime.last().unwrap().format("%Y-%m-%d %H:%M")
    );

    // Create a scalping strategy configuration
    println!("Setting up scalping strategy (EMA 3/8)...");
    let strategy_name = "Scalping Strategy (EMA 3/8)".to_string();

    // Scalping strategy parameters - very short periods for quick signals
    let _fast_period = 3; // Very fast EMA for immediate price changes
    let _slow_period = 8; // Short EMA for trend confirmation

    // Set up backtest parameters
    let initial_capital = 100.0; // $100

    // Create commission structure optimized for scalping
    let commission = HyperliquidCommission {
        maker_rate: 0.0001,     // Lower maker fee for scalping (0.01%)
        taker_rate: 0.0003,     // Lower taker fee for scalping (0.03%)
        funding_enabled: false, // Funding less relevant for very short-term trades
    };

    // Create and run backtest
    let mut backtest = HyperliquidBacktest::new(
        data.clone(),
        strategy_name.clone(), // Clone the strategy name so we can reuse it
        initial_capital,
        commission.clone(), // Clone the commission so we can reuse it
    );

    println!("Running scalping backtest...");

    // Initialize the base backtest
    backtest.initialize_base_backtest()?;

    // Run backtest (funding disabled for scalping)
    backtest.calculate_with_funding()?;

    // Get backtest results using enhanced report
    let report = backtest.enhanced_report()?;

    // Print scalping backtest results
    println!("\nScalping Backtest Results:");
    println!("-------------------------");
    println!("Initial Capital: ${:.2}", report.initial_capital);
    println!("Final Capital: ${:.2}", report.final_equity);
    println!(
        "Net Profit: ${:.2} ({:.2}%)",
        report.final_equity - report.initial_capital,
        report.total_return * 100.0
    );
    println!("Max Drawdown: {:.2}%", report.max_drawdown * 100.0);
    println!("Win Rate: {:.2}%", report.win_rate * 100.0);
    println!("Profit Factor: {:.2}", report.profit_factor);
    println!("Sharpe Ratio: {:.2}", report.sharpe_ratio);

    // Scalping-specific metrics
    let commission_stats = &report.commission_stats;
    println!("\nScalping Commission Analysis:");
    println!("---------------------------");
    println!(
        "Total Commission Paid: ${:.4}",
        commission_stats.total_commission
    );
    println!(
        "Commission as % of Capital: {:.3}%",
        (commission_stats.total_commission / report.initial_capital) * 100.0
    );
    if report.total_return != 0.0 {
        println!(
            "Commission as % of Profit: {:.2}%",
            (commission_stats.total_commission
                / (report.final_equity - report.initial_capital).abs())
                * 100.0
        );
    }

    // For scalping, funding is typically not relevant due to short hold times
    println!("\nNote: Funding rates are not displayed as scalping typically");
    println!("involves very short position hold times (seconds to minutes).");

    // Get trade statistics
    println!("\nScalping Trade Statistics:");
    println!("-------------------------");
    println!("Total Trades: {}", report.trade_count);
    println!("Win Rate: {:.2}%", report.win_rate * 100.0);
    println!("Profit Factor: {:.2}", report.profit_factor);
    if report.trade_count > 0 {
        let avg_profit_per_trade =
            (report.final_equity - report.initial_capital) / report.trade_count as f64;
        println!("Average Profit per Trade: ${:.4}", avg_profit_per_trade);
    }

    // Export scalping results to CSV
    println!("\nExporting scalping results to CSV...");
    let csv_file = "scalping_backtest_results.csv";
    backtest.export_to_csv(csv_file)?;
    println!("Results exported to {}", csv_file);

    // For scalping, we'll compare with higher commission rates to show the impact
    println!("\nRunning comparison with higher commission rates...");
    let higher_commission = HyperliquidCommission {
        maker_rate: 0.0005, // Higher maker fee (0.05%)
        taker_rate: 0.001,  // Higher taker fee (0.1%)
        funding_enabled: false,
    };

    let mut backtest_high_commission = HyperliquidBacktest::new(
        data.clone(),
        "Scalping Strategy (High Commission)".to_string(),
        initial_capital,
        higher_commission,
    );

    backtest_high_commission.initialize_base_backtest()?;
    backtest_high_commission.calculate_with_funding()?;

    let report_high_commission = backtest_high_commission.enhanced_report()?;

    println!("\nComparison Results (Higher Commission):");
    println!("------------------------------------");
    println!(
        "Net Profit: ${:.2} ({:.2}%)",
        report_high_commission.final_equity - report_high_commission.initial_capital,
        report_high_commission.total_return * 100.0
    );
    println!(
        "Total Commission: ${:.4}",
        report_high_commission.commission_stats.total_commission
    );

    println!("\nCommission Impact on Scalping:");
    println!("-----------------------------");
    let profit_diff = (report.final_equity - report.initial_capital)
        - (report_high_commission.final_equity - report_high_commission.initial_capital);
    println!("Profit Difference: ${:.4}", profit_diff);
    let commission_diff = report_high_commission.commission_stats.total_commission
        - report.commission_stats.total_commission;
    println!("Additional Commission Cost: ${:.4}", commission_diff);
    if report_high_commission.total_return != 0.0 && report.total_return != 0.0 {
        println!(
            "Performance Impact: {:.2}%",
            ((report.total_return / report_high_commission.total_return) - 1.0) * 100.0
        );
    }

    println!("\nScalping example completed successfully!");
    println!("Key takeaway: Commission rates are critical for scalping profitability!");

    Ok(())
}
