# Titanfx Project Analysis (Architecture + Data First)

## What this project is
Titanfx is a multi-pair FX modeling project built around `TITAN-NL v5.0`, with an end-to-end path from feature generation to training/backtest to live inference/execution.

Current scripts map to four workflows:
1. `build_titan_15m.py` — 15m dataset construction from M1 bars + macro + calendar.
2. `titan.py` — training/backtesting with dual-head signal + RealPnLLoss + triple-barrier targets.
3. `titan_live_2026.py` — standalone live-style calibration/backtest script.
4. `titan_mt5_live.py` — MT5-connected live execution loop.

---

## Architecture read: what is already strong

### Model stack in `titan.py`
The architecture combines:
- **SelfModifyingDeltaMemory** for stateful sequence memory,
- **ContinuumMemoryMLP** for multi-timescale representation,
- **MarketRegimeMemory** for cross-pair graph interaction,
- **Dual-head output** (`direction × gate`) for directional conviction + participation,
- **RealPnLLoss** for tradability-aware optimization,
- **Triple-barrier realized targets** for realistic horizon outcomes.

This is a strong design direction for FX: sequence memory + cross-asset context + cost-aware objective.

### Data design in `build_titan_15m.py`
Feature families are broad and useful:
- per-pair OHLCV + technical transforms,
- cross-pair relative strength + rolling correlation,
- macro context (commodities, yields, equities, CB rates),
- event/calendar encoding,
- time features.

The feature breadth is enough to support meaningful model improvements without needing a totally new architecture.

---

## Main blockers to a better model (architecture + data)

### 1) Train/live architecture drift
- `titan.py` and `titan_live_2026.py` both define architecture internals.
- Any small mismatch can degrade live behavior even if training metrics improve.

### 2) Feature contract instability
- Training and live feature builders are not fully unified.
- Live aliases + dynamic filling can create subtle distribution shifts.

### 3) Objective/target alignment still improvable
- Triple-barrier is strong, but static `(K_TP, K_SL, MAX_HOLD)` may be too rigid across regimes.
- RealPnLLoss is good, but its penalty mix should likely be regime-adaptive.

### 4) Representation efficiency
- The architecture has strong components, but no explicit mechanism for:
  - sparse feature selection under regime change,
  - uncertainty-aware gating,
  - pair-specific expert specialization.

---

## Recommended upgrade plan (focused on model quality)

## Phase A — Unify architecture + feature contract (highest ROI)
1. Extract one shared core package (`titan_core`) with:
   - model blocks,
   - loss,
   - feature schema contract,
   - inference adapter.
2. Make train/live scripts import the exact same classes.
3. Freeze a strict schema manifest:
   - ordered feature list,
   - dtype,
   - normalization policy,
   - missing-value policy.

**Expected gain:** removes hidden train/live mismatch and makes architecture experiments trustworthy.

## Phase B — Data quality upgrades for alpha signal
1. Add **regime tags** as explicit features (volatility state, trend state, risk-on/off proxy).
2. Replace some static rolling windows with **adaptive windows** (volatility-scaled lookbacks).
3. Enrich cross-pair structure with:
   - rolling lead/lag features,
   - pair-cluster factors,
   - residual/cointegration-style spreads.
4. Improve event handling with:
   - surprise standardization by indicator,
   - event decay curves (t+1..t+n impact),
   - pre/post-event regime flags.

**Expected gain:** better context representation and less feature noise.

## Phase C — Target/loss improvements
1. Make triple-barrier parameters regime-conditional:
   - wider barriers in high-vol,
   - tighter in low-vol.
2. Add secondary target head(s):
   - hit-probability (`P(TP before SL)`),
   - expected holding time.
3. Calibrate loss weights online:
   - `lambda_tc`, `lambda_cvar` adapted by realized turnover and drawdown.

**Expected gain:** tighter coupling between predicted signal and executable PnL.

## Phase D — Architecture experiments (controlled)
Run ablations one variable at a time:
1. **Gate head variants:** sigmoid gate vs sparsemax/entmax gate.
2. **Memory variants:** keep delta memory, test low-rank memory projection.
3. **Cross-pair block:** compare current graph attention vs pair-factor mixer.
4. **MoE head:** small mixture-of-experts per pair/regime.

Track with fixed splits and same transaction-cost model.

**Expected gain:** isolate real architectural improvements instead of noise.

---

## Concrete experiment matrix to start now

1. **Baseline lock**
   - Freeze current dataset and architecture as control run.
2. **Contract lock run**
   - Same model, but strict shared feature contract train/live.
3. **Regime-feature run**
   - Add only regime tags + adaptive windows.
4. **Adaptive-barrier run**
   - Add only regime-conditional barrier parameters.
5. **Gate-variant run**
   - Swap gate activation, keep everything else fixed.

Compare by:
- out-of-sample Sharpe,
- turnover-adjusted return,
- max drawdown,
- hit-rate by pair and regime,
- signal stability (flip rate).

---

## Best next step
If the goal is better model performance quickly:

**Do Phase A first (shared architecture + strict feature contract), then Phase B (regime-aware data upgrades).**

That gives the cleanest base for meaningful architecture improvements and avoids wasting time on experiments with hidden data/interface drift.
