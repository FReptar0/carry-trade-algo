# Continuous Volatility Scaling — Implementation Tracker

> Phase 1 Enhancement #8 from `docs/ENHANCEMENTS.md`
> Implemented: 2026-02-08

---

## Summary

Each tick, compares current ATR to the ATR recorded at entry time for every open position. When the volatility ratio exceeds a configurable threshold (default 1.5×), the system trims the position via partial close to restore the originally intended dollar-risk exposure.

This is a **defensive feature** — it prevents oversized risk when markets become volatile after entry.

---

## Configuration

Added to `RunnerConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_vol_scaling` | `bool` | `True` | Master switch |
| `vol_scaling_threshold` | `float` | `1.5` | ATR ratio that triggers a trim |
| `vol_scaling_cooldown_hours` | `int` | `24` | Minimum hours between trims per pair |
| `vol_scaling_min_position_pct` | `float` | `0.25` | Floor — never trim below 25% of original units |

---

## Position Dict Changes

Three new fields added to every position in `_strategy_positions[pair]`:

| Field | Type | Description |
|-------|------|-------------|
| `entry_atr` | `Optional[float]` | ATR(14) at entry time. `None` for legacy synced positions. |
| `original_units` | `int` | Units at first entry (before any vol trims) |
| `last_vol_trim_time` | `Optional[datetime]` | Timestamp of last vol trim (for cooldown enforcement) |

---

## Logic Flow

1. `_check_volatility_scaling(now)` runs each tick at step 11c (after stop-loss updates, before equity snapshot)
2. For each position with known `entry_atr`:
   - Compute `vol_ratio = current_atr / entry_atr`
   - Skip if `vol_ratio <= threshold` (1.5)
   - Skip if cooldown not elapsed (24h since last trim)
   - Target units = `original_units × entry_atr / current_atr`, floored at `original_units × 0.25`
   - Trim via `broker.close_position(pair, units=trim_units)`
   - Update position state + send WARNING Telegram alert
3. Synced positions (from broker reconciliation) estimate `entry_atr` from stop distance: `atr_est = (entry_price - stop_price) / atr_stop_mult`

---

## Telegram Display

- `/positions`: Shows `Vol X.Xx` per position. Warning emoji when >1.5x. Shows `Orig X,XXXu` if trimmed.
- `/status`: Appends `⚠️X.Xx` warning suffix on compact position lines when vol_ratio >1.5.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/engine/runner.py` | `RunnerConfig` (4 fields), `_promote_filled_order` (3 new pos fields), `_sync_positions` (3 fields + ATR estimation), `_check_volatility_scaling()` (115-line method), tick step 11c, `get_system_state()` computes vol_ratio |
| `src/ops/telegram_bot.py` | `/positions` vol ratio + warning, `/status` vol warning suffix |
| `tests/unit/test_runner.py` | 13 new `TestVolatilityScaling` tests |
| `tests/unit/test_telegram_bot.py` | 8 new `TestVolatilityScalingDisplay` tests |
| `docs/ENHANCEMENTS.md` | Item #8 marked ✅ |

---

## Test Coverage

- **13 runner tests** (`TestVolatilityScaling`): trim logic, threshold, cooldown, floor, disabled, missing data, broker failure, state updates, alerts, entry_atr storage, sync estimation
- **8 Telegram bot tests** (`TestVolatilityScalingDisplay`): vol ratio display, warning emoji, original units, missing fields graceful handling
- **Full suite**: 598 passed, 11 skipped, 0 failures (expected)

---

## Checklist

- [x] Research: Read current position sizing, ATR tracking, and scale-out logic
- [x] Design: Define vol scaling config, threshold, trim logic, and cooldown rules
- [x] Implement: Add entry_atr tracking to position dict and sync
- [x] Implement: Build `_check_volatility_scaling()` in runner.py
- [x] Implement: Wire vol scaling into the tick loop
- [x] Implement: Add vol scaling info to Telegram /positions and /status
- [x] Implement: Add vol scaling alerts
- [x] Write runner tests (13 tests in TestVolatilityScaling)
- [x] Write Telegram bot tests (8 tests in TestVolatilityScalingDisplay)
- [x] Run full test suite — all passing
- [x] Update docs/ENHANCEMENTS.md to mark complete
- [x] Create this tracker document
- [ ] Commit all changes
- [ ] Push to GitHub
- [ ] Deploy to EC2
- [ ] Verify logs on EC2
