# Hyperliquid Scalping Bot Development Playbook

## 0 · Bird’s-eye architecture — what we’re ultimately wiring together

```
┌────────────┐     WS::Book/Trades     ┌────────────┐
│ HL Gateway │ ──────────────────────► │  DataBus   │   (Tokio mpsc)
└────────────┘ ◄────────────────────── └────────────┘
       ▲                                   │
       │      REST::Exchange / Info        ▼
┌────────────┐                      ┌──────────────┐
│ Strategy    │  async state-loop   │  Risk/MM     │
│ (Scalper)   │  orders  ◄──────────┤  Guards      │
└────────────┘                      └──────┬───────┘
                                           │    snapshots→ S3/Parquet
                                           ▼
                                 ┌──────────────────┐
                                 │ Back-test engine │
                                 └──────────────────┘
                                           ▼
                                 Grafana / Prometheus
```

*Everything sits in **one cargo workspace** so live-trading and back-tests share the exact same `strategy` crate—zero logic drift.*

## 1 · Environment bootstrap (10 min)

| Step                         | Command / crate                                                                          | Notes                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------- |
| 1. Install stable tool-chain | `rustup default stable`                                                                  | keep CI on the same minor               |
| 2. Workspace skeleton        | `cargo new --vcs git hl_scalper`                                                         |                                         |
| 3. Core deps                 | `cargo add hyperliquid_rust_sdk tokio polars rayon clap dotenvy tracing anyhow`          |                                         |
| 4. Verify SDK version        | `cargo search hyperliquid_rust_sdk` → **0.6.0** is current on crates.io ([crates.io](https://crates.io/crates/hyperliquid_rust_sdk?utm_source=chatgpt.com)) | |
| 5. Secrets outside git       | `.env : HL_PK=<hex-priv-key>`                                                            | Claude can auto-generate `.env.example` |

## 2 · Data interfaces (live + historical)

### 2.1 Realtime feed

* Use the SDK’s ready-made binaries (`ws_l2_book.rs`, `ws_bbo.rs`, `ws_trades.rs`) as reference.
* For production latency, subscribe only to `BBO` or `Trades` and reconstruct micro-price locally.

### 2.2 Historical back-fill

| Granularity          | API / bucket                                    | Usage                                   |
| -------------------- | ----------------------------------------------- | --------------------------------------- |
| Candles ≤ 5 000 rows | `Info::candle_snapshot`                         | simple param sweeps                     |
| Tick/L2              | Public S3 `hyperliquid-archive/market_data/...` | replay for queue-position modelling     |
| Custom               | persist your WS stream → Parquet                | gives *identical* schema for live & sim |

Claude-assist ❯ auto-generate a daily **downloader task** and a Polars converter.

## 3 · Strategy kernel — “slot-runner” scalp

1. **Trigger**
   * order-book imbalance > 60 %
   * spread < 2 bps
   * micro-momentum (mid − 15 s EMA) > 0
2. **Entry** – post-only limit *one* tick inside spread.
3. **Size** – Kelly fraction, but hard-cap ΔUSD ≤ 0.5 % equity.
4. **Exit** – +2 ticks **or** 5 s timeout **or** ≥ 4 tick adverse move.
5. **Throttle** – ≤ 10 cancels / second (HL hard-limit).

Claude-assist ❯ write quick-check property tests ensuring every generated order respects (2)–(5).

## 4 · Execution layer snippets

```rust
// whole file: src/bin/place_and_cancel.rs
use chrono::Utc;
use ethers::signers::LocalWallet;
use hyperliquid_rust_sdk::{
    Exchange,
    types::{OrderInput, Verb},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let wallet: LocalWallet = std::env::var("HL_PK")?.parse()?;
    let ex = Exchange::new(None).await?;          // main-net

    // mid-price
    let mid = ex.info().mid_px("ETH").await?;

    // inside-spread bid
    let px  = mid * 0.9998;
    let qty = 0.25;
    let order = OrderInput::perp("ETH".into(), Verb::Bid, qty, Some(px), None, None);
    ex.place_order(Arc::new(wallet.clone()), vec![order], None).await?;

    // global cancel in 5 s
    let t_cancel = (Utc::now().timestamp_millis() + 5_000) as u64;
    ex.schedule_cancel(Arc::new(wallet), Some(t_cancel)).await?;
    Ok(())
}
```

*Pass a test-net URL to `Exchange::new(Some(…))` for sandboxing.*

Claude-assist ❯ lint, format and generate missing error-handling branches.

## 5 · Back-test engine (event-driven)

* **DataFrame engine** – `polars::lazyframe` ⟶ 10 M candles/s on laptop.
* **Fill model** – two-tier:
  → “Touches price ⇒ filled” for fast sweeps.
  → L2 replay when you need queue-position realism.
* **Parallel param grid** – wrap each Backtest in `rayon::scope`; avoid Tokio in sim phase.
* **Metrics to parquet** – Grafana picks up PnL, Sharpe, hit-rate, hold-time.

Claude-assist ❯ auto-create a Jupyter notebook that pulls the Parquet and renders equity curves.

## 6 · Operational checklist

| Layer            | Check                                                     |
| ---------------- | --------------------------------------------------------- |
| **Connectivity** | re-subscribe on `isSnapshot=true` after WS kick           |
| **Clock skew**   | `Info::meta().serverTime` vs `Utc::now()`; alert > 150 ms |
| **Cancels**      | prefer `schedule_cancel`; 10 triggers/day limit           |
| **Monitoring**   | Prometheus counters for fills, rejects, latency buckets   |
| **Recovery**     | on restart call `Info::get_orders` then reconcile         |

Claude-assist ❯ generate Prometheus exporter scaffolding with `metrics` crate.

## 7 · Safety rails (do **not** skip)

* **Dead-man**: refresh `schedule_cancel` every 4 s (half your cancel window).
* **Equity stop**: auto-liquidate & exit if drawdown > 2 % intraday.
* **Vault segregation**: trade from a dedicated HL vault, keep bulk funds cold.
* **Alerting**: PagerDuty on missing WS heartbeat > 3 s or dead-man failures > 2.

Claude-assist ❯ set conditional reminders or automations for these thresholds.

## 8 · Deployment & CI

1. **Unit + property tests** – run on `cargo test`.
2. **Back-tests** – nightly GitHub Actions matrix on latest tick dump. Fail build if Sharpe drops > 20 % vs 7-day median.
3. **Container** – static `scratch` image (`musl + strip`).
4. **Canary** – deploy to small-sized sub-account; promote automatically after +100 trades & positive PnL.

Claude-assist ❯ write the GitHub Actions workflow, including secret scanning.

## 9 · Smart next extensions (speculative ⚡)

| Idea                                                                                    | Why it’s interesting                    |
| --------------------------------------------------------------------------------------- | --------------------------------------- |
| **Sub-µs latency** – isolate strategy core, pin CPU, `TCP_NODELAY`.                     | Might add 0.4 bps edge on liquid pairs. |
| **Funding curve alpha** – combine HL funding (`Info::funding_history`) with short-skew. | Captures carry-trade premium.           |
| **Adaptive cancel window** – scale `schedule_cancel` by recent micro-vol.               | Cuts reject rate during chop.           |
| **Cross-venue hedging** – delta-match on CEX via WS; careful with bridge latency.       | Reduces inventory VaR.                  |

## Quick links you’ll keep opening

```
SDK repo      https://github.com/hyperliquid-dex/hyperliquid-rust-sdk
API docs      https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
SDK crate     https://crates.io/crates/hyperliquid_rust_sdk
Historical S3 https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data
```

## How Claude can help day-to-day

| Task                      | Claude prompt                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------- |
| Generate a new param grid | “Write me a Rayon loop that sweeps ε=[1–5 bps] & hold=[200–800 ms] over *backtest.rs*”     |
| Inspect Grafana anomaly   | “Explain the spike in cancel rejects at 14:02 UTC using last 200 WS messages (paste below).” |
| Draft post-mortem         | “Summarise today’s PnL deviation vs model; include links to Prometheus graphs.”              |

**You now have one definitive playbook—architecture, code scaffolds, back-tester, ops, and ideas for alpha—in the exact order a Claude agent (or any disciplined dev) can execute without ambiguity. Plug in your alpha and ship.**
