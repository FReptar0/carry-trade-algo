# Real Money Readiness — Recommendations

> Created: 2026-02-17
> Context: V3 is live on OANDA practice account, day 14/34 of validation protocol.
> Goal: Document everything needed before trading real capital.

---

## Component Status (as of Feb 2026)

| Component | Connected? | Behavior |
|-----------|-----------|----------|
| Regime detector | ✅ Partial | Scales position size only (0.25x–1.0x) — does NOT block entries |
| News blackout | ✅ Full | Hard blocks new entries ±2h around high-impact events |
| ML bandit | ⚠️ Shadow | Observing + learning, auto-activates at 50 trades — not blocking yet |
| Correlation monitor | ✅ Partial | Reduces sizing when correlated — no hard cap on simultaneous positions |
| Circuit breakers | ✅ Full | Halts at daily 3%, weekly 7%, DD 20% |
| Dynamic sizer | ✅ Full | ATR + Kelly + regime + drawdown-aware |

---

## Recommendations

### MUST DO before real money

#### 1. Regime as hard entry gate
**Status**: Not implemented. Regime only affects sizing.
**What to do**: Block entries when `CompositeRegime == RANGE_HIGH_VOL` or `RANGE_LOW_VOL`.

```
TREND_LOW_VOL   → trade normally (full size)
TREND_HIGH_VOL  → trade at reduced size  ← already working
RANGE_LOW_VOL   → BLOCK entry            ← currently enters at 25% size
RANGE_HIGH_VOL  → BLOCK entry            ← currently enters at 25% size
```

Change location: `runner.py` — add check before `_open_position()` call.

#### 2. Lower ML bandit activation threshold
**Status**: Requires 50 trades + 10% advantage. At 1–2 trades/pair/month this takes 4–8 months.
**What to do**: Lower `min_trades` from 50 → 20 in `ShadowEvaluator`. Keep the 10% advantage gate.
Change location: `runner.py:188` — `ShadowEvaluator(min_trades=20, min_advantage=0.1)`.

#### 3. Verify live swap rates are fetched each tick
**Status**: Unknown — runner has `swap_long_default: 0.0` fallback.
**What to do**: Confirm `OandaBroker.get_swap_rates()` is called per tick and results override
defaults. If swap rates are wrong, the carry rationale is broken.
Risk: Entering trades with negative carry that appear positive.

#### 4. Define drawdown recovery protocol
**Status**: Circuit breaker halts at 20% DD but no documented recovery procedure.
**What to do**: Add to runner config:

```
10% DD → new position sizes × 0.5
15% DD → halt new entries, manage existing only
20% DD → full halt (circuit breaker already does this)
Post-halt re-entry → only when equity recovers above -10% DD level
```

#### 5. Complete a clean second forward test (60–90 days)
**Status**: Current 34-day protocol has a bug period (Feb 10–13) and balance restore.
**What to do**: After day 34, run a second clean forward test with all fixes live.
**Minimum trades needed for statistical significance**: 50–100 completed trades.
The 71% win rate at 17 trades has ±20% confidence interval — not trustworthy yet.

---

### HIGH IMPACT — implement before scaling up

#### 6. Cross-asset macro filter (VIX / DXY)
**Status**: `FeatureExtractor` fetches VIX, DXY, 10Y yield — used only as bandit features.
**What to do**: Add as hard entry gates:

| Condition | Action |
|-----------|--------|
| VIX > 25 | Block new entries (risk-off = carry unwinds) |
| DXY weekly trend declining | Reduce USD/JPY sizing |
| 10Y yield falling sharply | Flag JPY strengthening risk |

Historical validation: Would have blocked Feb 10 bug-period entries and most major carry unwinds.

#### 7. BOJ meeting 48h blackout
**Status**: `EconomicCalendar` covers high-impact events at ±2h. Not enough for BOJ.
**What to do**: BOJ surprise decisions (Jan 2016, Jul 2024) cause 3–5% moves. Add dedicated
48h blackout around all BOJ meetings. Worth hardcoding the annual BOJ meeting schedule.

#### 8. Weekend gap protection
**Status**: No protection. Last signal Friday ~21:00 UTC. Next tick Monday 00:00 UTC.
**What to do**: Close or halve all positions before Friday 20:00 UTC. Re-evaluate Monday open.
Cost: Some lost swap income (2 days). Benefit: Eliminates gap risk (Jan 2019 flash crash was Sunday night).

---

### MEDIUM IMPACT — within 3 months of going live

#### 9. Correlation hard cap
**Status**: `CorrelationMonitor` reduces sizing but doesn't cap simultaneous positions.
**What to do**: When rolling 10-day pairwise correlation > 0.85 between any two open pairs,
block new entry regardless of signal.
Reason: 6 JPY pairs at 0.90 correlation = 1 trade with 6× effective leverage.

#### 10. Per-pair parameter optimization
**Status**: All 6 pairs share identical V3 params (50/200 MA, 3× ATR stop).
**What to do**: Run Optuna optimizer per pair, not per regime. Key differences:

| Pair | Characteristic | Recommendation |
|------|---------------|----------------|
| USD/JPY | Most liquid, tight spread | Can tighten ATR stop to 2.5× |
| GBP/JPY | Highest volatility | Widen to 4× ATR, smaller size |
| NZD/JPY | Commodity-linked, gappier | Add VIX gate specifically |
| AUD/JPY | Similar to NZD/JPY | Separate optimization needed |

#### 11. Monthly formal review cadence
**Status**: No automated monthly report.
**What to do**: Build `scripts/monthly_review.py` covering:
- Live vs backtest return comparison
- Regime distribution (% time TREND vs RANGE)
- Swap income vs price PnL attribution per pair
- Bandit shadow stats (SKIP rate, counterfactual PnL)
- Correlation heatmap across the month

---

## Capital Sizing for Real Money

Do NOT mirror the practice account. Start small to validate execution quality on real accounts
(wider spreads, worse fills, stop slippage during fast markets).

| Phase | Capital | Condition to proceed |
|-------|---------|---------------------|
| Month 1–2 | $5,000–10,000 | After clean 60-day forward test |
| Month 3–4 | $20,000–25,000 | Live Sharpe ≥ 70% of backtest Sharpe |
| Month 6+ | Scale to full | 100+ completed trades, DD never exceeded 10% live |

---

## Full Readiness Checklist

### Before any real money
- [ ] Regime hard gate implemented and tested (RANGE → block entry)
- [ ] Bandit threshold lowered to 20 trades
- [ ] Live swap rate fetch verified
- [ ] Drawdown recovery protocol documented and coded
- [ ] Clean 34-day protocol completed (no bug periods)
- [ ] Second 60-90 day clean forward test completed
- [ ] 50+ completed trades on record

### Before scaling beyond $10k
- [ ] VIX/DXY macro filter implemented
- [ ] BOJ meeting 48h blackout implemented
- [ ] Weekend gap protection implemented
- [ ] Live Sharpe ≥ 70% of backtest Sharpe
- [ ] Correlation hard cap implemented
- [ ] Monthly review script operational

### Before full allocation
- [ ] 100+ completed trades
- [ ] Max live drawdown never exceeded 10%
- [ ] Per-pair parameter optimization done
- [ ] 6+ months continuous operation without abort

---

## Implementation Priority Order

```
NOW (protocol days 14–34):
  1. Regime hard gate           ← highest impact, one-day implementation
  2. Bandit threshold → 20      ← one-line change
  3. Verify swap rate fetch     ← audit only

AFTER protocol day 34:
  4. Define + code DD recovery protocol
  5. Start second 60-day forward test
  6. VIX/DXY macro filter
  7. BOJ 48h blackout
  8. Weekend gap protection

AFTER second forward test:
  9.  Correlation hard cap
  10. Per-pair optimization
  11. Monthly review script
  12. Start real money at $5-10k
```

---

## Key Risks to Monitor Always

1. **BOJ surprise** — Any unscheduled BOJ statement or rate decision. All 6 pairs move simultaneously.
2. **Correlated drawdown** — When all 6 JPY pairs are open and risk-off hits, effective drawdown is 6× single-pair.
3. **Swap rate flip** — If BOJ raises rates, USD/JPY and others lose their carry advantage. The swap gate catches this but only on the next tick.
4. **Execution degradation** — Real fills ≠ practice fills. Monitor actual vs expected slippage from day 1.
5. **Regime misclassification** — The regime detector uses ADX + ATR on 300 hourly bars. During fast regime transitions it can lag by several hours.
