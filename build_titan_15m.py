"""
=============================================================================
 Titan 15M Forex Dataset Builder
 ────────────────────────────────
 Fuses M1 forex data + daily context sources into a 15-minute dataset
 optimised for the TITAN-NL model.

 Output : Titan15M_Dataset.csv (~335k rows × ~280 cols)
=============================================================================
"""

import os, sys, warnings, re, json, math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def _resolve_base_dir() -> Path:
    # Priority: explicit env var -> common Kaggle paths -> current working directory
    if os.getenv("TITAN_DATA_BASE"):
        return Path(os.getenv("TITAN_DATA_BASE")).expanduser().resolve()

    kaggle_candidates = [
        Path('/kaggle/working'),
        Path('/kaggle/input/titanfx'),
        Path('/kaggle/input'),
    ]
    for c in kaggle_candidates:
        if c.exists():
            return c
    return Path.cwd()


BASE = _resolve_base_dir()
OUT  = Path(os.getenv("TITAN_OUT_PATH", str(Path('/kaggle/working/Titan15M_Dataset.csv' if Path('/kaggle/working').exists() else BASE / 'Titan15M_Dataset.csv'))))
EVENT_ENCODER_OUT = BASE / "event_label_encoder_15m.json"

DATE_START = "2024-01-01"
DATE_END   = "2024-12-31"

PAIRS = {
    "EURUSD": BASE / "EURUSD_M1_2010-2024.csv",
    "GBPUSD": BASE / "GBPUSD_M1_2010-2024.csv",
    "AUDUSD": BASE / "AUDUSD_M1_2010-2024.csv",
    "USDJPY": BASE / "USDJPY_M1_2010-2024.csv",
}

import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

def log(msg): print(f"[Titan15M] {msg}")

def pct(s: pd.Series, n: int = 1) -> pd.Series:
    return np.log(s / s.shift(n))

def sma(s, n):  return s.rolling(n, min_periods=1).mean()
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

# =============================================================================
# STAGE 1 - Forex OHLCV & Technicals
# =============================================================================
def stage_1_forex_15m():
    log("Stage 1 ▸ Aggregating M1 → 15M bars + computing indicators …")
    frames = {}

    for pair, path in PAIRS.items():
        log(f"  ⤷ {pair} ({path.name}) — reading in chunks …")
        chunks = []
        for chunk in pd.read_csv(path, parse_dates=["datetime"], chunksize=2_000_000):
            chunk = chunk.set_index("datetime")
            bar15 = chunk.resample("15min").agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["close"])
            chunks.append(bar15)

        df = pd.concat(chunks)
        df = df.groupby(df.index).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        })

        p = pair
        # KEEP OHLCV
        df = df.rename(columns={"open": f"{p}_Open", "high": f"{p}_High", 
                                "low": f"{p}_Low", "close": f"{p}_Close", "volume": f"{p}_Volume"})
        
        c = df[f"{p}_Close"]
        h = df[f"{p}_High"]
        l = df[f"{p}_Low"]
        o = df[f"{p}_Open"]
        v = df[f"{p}_Volume"]

        # Log returns
        df[f"{p}_ret_1"]  = pct(c, 1)
        df[f"{p}_ret_2"]  = pct(c, 2)
        df[f"{p}_ret_4"]  = pct(c, 4)
        df[f"{p}_ret_8"]  = pct(c, 8)
        df[f"{p}_ret_12"] = pct(c, 12)

        # Structure
        df[f"{p}_range"] = h - l
        df[f"{p}_body"]  = (c - o).abs()
        df[f"{p}_upper_wick"] = h - pd.concat([c, o], axis=1).max(axis=1)
        df[f"{p}_lower_wick"] = pd.concat([c, o], axis=1).min(axis=1) - l

        # Volatility
        df[f"{p}_atr_14"] = atr(h, l, c, 14)
        df[f"{p}_vol_std_20"] = pct(c).rolling(20, min_periods=1).std()

        # Trend
        df[f"{p}_sma_20_slope"] = sma(c, 20).diff() / sma(c, 20).shift(1).replace(0, np.nan)
        df[f"{p}_ema_cross"] = (ema(c, 12) - ema(c, 26)) / c

        # Momentum
        df[f"{p}_rsi_14"] = rsi(c, 14)
        df[f"{p}_roc_4"] = c.diff(4) / c.shift(4).replace(0, np.nan)
        
        frames[pair] = df
        log(f"    ✓ {pair}: {len(df)} bars, {df.shape[1]} columns")

    merged = frames["EURUSD"]
    for p in ["GBPUSD", "AUDUSD", "USDJPY"]:
        merged = merged.join(frames[p], how="inner")

    return merged

# =============================================================================
# STAGE 2 - Cross-Pair Relationships
# =============================================================================
def stage_2_cross_pair(merged):
    log("Stage 2 ▸ Cross-pair relative strength & factor models …")
    
    # Relative Strength
    merged["rs_EURUSD_GBPUSD"] = merged["EURUSD_ret_1"] - merged["GBPUSD_ret_1"]
    merged["rs_EURUSD_AUDUSD"] = merged["EURUSD_ret_1"] - merged["AUDUSD_ret_1"]
    merged["rs_GBPUSD_AUDUSD"] = merged["GBPUSD_ret_1"] - merged["AUDUSD_ret_1"]
    
    # Rolling Correlations (12 bars = 3 hours)
    for pair1, pair2 in [("EURUSD", "GBPUSD"), ("EURUSD", "AUDUSD"), ("EURUSD", "USDJPY")]:
        merged[f"corr_{pair1}_{pair2}"] = merged[f"{pair1}_ret_1"].rolling(12, min_periods=4).corr(merged[f"{pair2}_ret_1"])

    # USD Factor Proxy
    usd_eur = -merged["EURUSD_ret_1"]
    usd_gbp = -merged["GBPUSD_ret_1"]
    usd_aud = -merged["AUDUSD_ret_1"]
    usd_jpy = merged["USDJPY_ret_1"]
    
    merged["usd_factor"] = (usd_eur + usd_gbp + usd_aud + usd_jpy) / 4.0

    # Divergence
    for p, inv_mult in [("EURUSD", -1), ("GBPUSD", -1), ("AUDUSD", -1), ("USDJPY", 1)]:
        pair_usd_proxy = merged[f"{p}_ret_1"] * inv_mult
        merged[f"{p}_usd_divergence"] = pair_usd_proxy - merged["usd_factor"]

    return merged

# =============================================================================
# STAGE 3 - Macro / Global Features
# =============================================================================
def stage_3_macro(merged):
    log("Stage 3 ▸ Incorporating daily macro features …")
    
    # Commodities
    commodities = pd.read_csv(BASE / "commodities_dataset.csv", parse_dates=["Date"])
    commodities = commodities.set_index("Date").sort_index()
    cols_map = {
        "Gold_('Close', 'GC=F')": "gold",
        "Crude_Oil_WTI_('Close', 'CL=F')": "wti"
    }
    commodities = commodities[list(cols_map.keys())].rename(columns=cols_map).replace(0, np.nan)
    rets_comm = pd.DataFrame(index=commodities.index)
    rets_comm["gold_ret"] = pct(commodities["gold"])
    rets_comm["wti_ret"] = pct(commodities["wti"])

    # Indices (S&P 500)
    indices = pd.read_csv(BASE / "2008_Globla_Markets_Data.csv")
    indices["Date"] = pd.to_datetime(indices["Date"])
    sp500 = indices[indices["Ticker"] == "^GSPC"].set_index("Date").sort_index()
    rets_comm["sp500_ret"] = pct(sp500["Close"].replace(0, np.nan))

    # Yields
    yields = pd.read_csv(BASE / "yields.csv")
    yields["date"] = pd.to_datetime(yields["time"], unit="ms", errors="coerce").dt.normalize()
    yields = yields.dropna(subset=["date"]).set_index("date").sort_index().replace(0.0, np.nan)
    y_out = pd.DataFrame(index=yields.index)
    if "US02" in yields.columns: y_out["yield_US02"] = yields["US02"]
    if "US10" in yields.columns: y_out["yield_US10"] = yields["US10"]
    if "US02" in yields.columns and "US10" in yields.columns:
        y_out["yield_US2s10s_slope"] = yields["US10"] - yields["US02"]
    for c in y_out.columns:
        y_out[f"{c}_chg"] = y_out[c].diff()
        
    # Central Bank Rates
    df_cb = pd.read_csv(BASE / "Top8 CB IR.csv")
    q_to_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
    df_cb["date"] = df_cb.apply(lambda r: pd.Timestamp(year=int(r["Year"]),
                                                  month=q_to_month.get(r["Quarter"], 1),
                                                  day=1), axis=1)
    df_cb = df_cb.set_index("date").sort_index()
    banks = ["Fed", "ECB", "BoE", "BoJ", "PBoC", "RBA", "BoC", "SNB"]
    for b in banks:
        df_cb[b] = pd.to_numeric(df_cb[b].replace("N/A", np.nan), errors="coerce")

    daily_idx = pd.date_range(DATE_START, DATE_END, freq="D")
    df_cb = df_cb[banks].reindex(daily_idx).ffill().bfill()
    
    out_cb = pd.DataFrame(index=df_cb.index)
    if "Fed" in df_cb.columns:
        for other, label in [("ECB", "Fed_ECB"), ("BoE", "Fed_BoE"),
                              ("BoJ", "Fed_BoJ"), ("RBA", "Fed_RBA")]:
            if other in df_cb.columns:
                out_cb[f"rate_diff_{label}"] = df_cb["Fed"] - df_cb[other]
    for b in banks:
        out_cb[f"cb_{b}_qoq_chg"] = df_cb[b].diff(90)

    # Reindex and forward fill to 15M exactly
    macro = pd.concat([rets_comm, y_out, out_cb], axis=1)
    macro = macro[~macro.index.duplicated()]

    reindexed = macro.reindex(merged.index, method="ffill")
    merged = merged.join(reindexed)
    
    return merged

# =============================================================================
# STAGE 4 - Forex Factory Calendar
# =============================================================================
def stage_4_calendar(merged):
    log("Stage 4 ▸ Calendar exact event features …")
    df = pd.read_csv(BASE / "forex_factory_cache.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["DateTime"])
    df = df.sort_values("DateTime")
    
    target_ccy = ["USD", "EUR", "GBP", "AUD", "JPY"]
    df = df[df["Currency"].isin(target_ccy)].copy()
    
    def parse_num(x):
        if pd.isna(x): return np.nan
        x = str(x).replace("%", "").replace("K", "e3").replace("M", "e6").replace("B", "e9")
        x = re.sub(r"[^\d.\-e+]", "", x)
        try: return float(x)
        except: return np.nan

    df["actual_n"]   = df["Actual"].apply(parse_num)
    df["forecast_n"] = df["Forecast"].apply(parse_num)
    df["surprise"]   = df["actual_n"] - df["forecast_n"]

    df["surprise_zscore"] = df.groupby("Event")["surprise"].transform(lambda x: (x - x.mean()) / x.std())
    df["surprise_zscore"] = df["surprise_zscore"].fillna(0)

    impact_map = {"High Impact Expected": 3, "Medium Impact Expected": 2, "Low Impact Expected": 1, "Non-Economic": 0}
    df["impact_level"] = df["Impact"].map(impact_map).fillna(0)

    # Filter only High Impact for "time to / from"
    high_impacts = np.sort(df[df["impact_level"] == 3]["DateTime"].dropna().unique())

    merged_idx = merged.index.values
    if len(high_impacts) > 0:
        high_impacts_dt64 = np.array(high_impacts, dtype='datetime64[ns]')
        
        pos = np.searchsorted(high_impacts_dt64, merged_idx, side='left')
        next_pos = np.clip(pos, 0, len(high_impacts_dt64)-1)
        next_dt = high_impacts_dt64[next_pos]
        mins_to = (next_dt - merged_idx).astype('timedelta64[m]').astype(float)
        mins_to[pos == len(high_impacts_dt64)] = 999999
        
        prev_pos = np.clip(pos - 1, 0, len(high_impacts_dt64)-1)
        prev_dt = high_impacts_dt64[prev_pos]
        mins_since = (merged_idx - prev_dt).astype('timedelta64[m]').astype(float)
        mins_since[pos == 0] = 999999
        
        merged["mins_to_high_event"] = mins_to
        merged["mins_since_high_event"] = mins_since
    else:
        merged["mins_to_high_event"] = 999999
        merged["mins_since_high_event"] = 999999

    df_grouped = df.groupby("DateTime").agg({
        "impact_level": "max",
        "surprise_zscore": lambda x: x.abs().max(),
        "Event": lambda x: list(x)
    })

    top_events = ["CPI", "NFP", "FOMC", "GDP", "PMI", "Retail_Sales", "Unemployment", "Interest_Rate", "Employment", "Manufacturing"]
    
    for ev in top_events:
        df_grouped[f"event_{ev}"] = 0

    def categorize_events(events_list):
        res = {k: 0 for k in top_events}
        for e in events_list:
            e_upper = str(e).upper()
            if "CPI" in e_upper or "CONSUMER PRICE INDEX" in e_upper: res["CPI"] = 1
            if "NON-FARM" in e_upper or "NFP" in e_upper: res["NFP"] = 1
            if "FOMC" in e_upper: res["FOMC"] = 1
            if "GDP" in e_upper or "GROSS DOMESTIC PRODUCT" in e_upper: res["GDP"] = 1
            if "PMI" in e_upper: res["PMI"] = 1
            if "RETAIL SALES" in e_upper: res["Retail_Sales"] = 1
            if "UNEMPLOYMENT" in e_upper: res["Unemployment"] = 1
            if "RATE" in e_upper or "CASH RATE" in e_upper or "FUNDS RATE" in e_upper: res["Interest_Rate"] = 1
            if "EMPLOYMENT" in e_upper or "PAYROLL" in e_upper: res["Employment"] = 1
            if "MANUFACTURING" in e_upper: res["Manufacturing"] = 1
        return pd.Series(res)
    
    cats = df_grouped["Event"].apply(categorize_events)
    for col in cats.columns:
        df_grouped[f"event_{col}"] = cats[col]
        
    df_grouped = df_grouped.drop(columns=["Event"])
    
    reindexed_events = df_grouped.reindex(merged.index)
    reindexed_events["impact_level"] = reindexed_events["impact_level"].fillna(0)
    reindexed_events["surprise_zscore"] = reindexed_events["surprise_zscore"].fillna(0)
    
    merged = merged.join(reindexed_events)
    merged["recent_surprise"] = merged["surprise_zscore"].ewm(alpha=0.1, ignore_na=True).mean()

    return merged

# =============================================================================
# STAGE 5 - Session & Time
# =============================================================================
def stage_5_time(merged):
    log("Stage 5 ▸ Time constraints & session overlap …")
    
    hour = merged.index.hour
    mins = merged.index.minute
    hour_decimal = hour + mins / 60.0
    
    merged["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24)
    merged["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24)
    
    dow = merged.index.dayofweek
    merged["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    merged["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    merged["session_tokyo"]   = np.where((hour >= 0) & (hour < 9), 1, 0).astype(np.int8)
    merged["session_london"]  = np.where((hour >= 7) & (hour < 16), 1, 0).astype(np.int8)
    merged["session_ny"]      = np.where((hour >= 13) & (hour < 22), 1, 0).astype(np.int8)
    merged["session_overlap"] = np.where((hour >= 13) & (hour < 16), 1, 0).astype(np.int8)
    
    merged["rollover_window"] = np.where((hour == 21) | (hour == 22), 1, 0).astype(np.int8)

    return merged



# =============================================================================
# STAGE 5B - Regime Features (Model-Quality Upgrades)
# =============================================================================
def stage_5b_regimes(merged):
    log("Stage 5B ▸ Adding regime-aware market state features …")

    pair_ret_cols = [f"{p}_ret_1" for p in PAIRS.keys() if f"{p}_ret_1" in merged.columns]
    if pair_ret_cols:
        fx_mean_ret = merged[pair_ret_cols].mean(axis=1)
        fx_dispersion = merged[pair_ret_cols].std(axis=1).fillna(0)
    else:
        fx_mean_ret = pd.Series(0.0, index=merged.index)
        fx_dispersion = pd.Series(0.0, index=merged.index)

    fx_vol_20 = fx_mean_ret.rolling(20, min_periods=5).std().fillna(0)
    vol_q = fx_vol_20.rolling(20 * 5, min_periods=20).rank(pct=True).fillna(0.5)
    merged["regime_volatility_percentile"] = vol_q
    merged["regime_high_vol"] = (vol_q > 0.67).astype(np.int8)
    merged["regime_low_vol"] = (vol_q < 0.33).astype(np.int8)

    fx_trend_ema_fast = fx_mean_ret.ewm(span=12, adjust=False).mean()
    fx_trend_ema_slow = fx_mean_ret.ewm(span=48, adjust=False).mean()
    trend_strength = (fx_trend_ema_fast - fx_trend_ema_slow).fillna(0)
    merged["regime_trend_strength"] = trend_strength
    merged["regime_trending"] = (trend_strength.abs() > trend_strength.rolling(96, min_periods=24).std().fillna(0)).astype(np.int8)

    if {"sp500_ret", "gold_ret"}.issubset(merged.columns):
        risk_proxy = (merged["sp500_ret"] - merged["gold_ret"]).fillna(0)
    elif "sp500_ret" in merged.columns:
        risk_proxy = merged["sp500_ret"].fillna(0)
    else:
        risk_proxy = pd.Series(0.0, index=merged.index)
    merged["regime_risk_proxy"] = risk_proxy
    merged["regime_risk_on"] = (risk_proxy.ewm(span=12, adjust=False).mean() > 0).astype(np.int8)

    merged["regime_cross_pair_dispersion"] = fx_dispersion
    return merged

# =============================================================================
# STAGE 6 - Targets & Leakage Processing
# =============================================================================
def stage_6_targets(df):
    log("Stage 6 ▸ Deriving exactly-aligned targets …")
    
    for pair in PAIRS.keys():
        c = df[f"{pair}_Close"]
        
        df[f"target_{pair}_ret_1"] = pct(c, 1).shift(-1)
        df[f"target_{pair}_ret_4"] = pct(c, 4).shift(-4)
        df[f"target_{pair}_ret_12"] = pct(c, 12).shift(-12)

    return df

# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = datetime.now()
    log(f"Starting NEW Titan 15M build at {t0:%Y-%m-%d %H:%M:%S}")
    log(f"Data base path: {BASE}")
    log(f"Output path: {OUT}")

    missing = [f"{pair}:{path}" for pair, path in PAIRS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pair CSV files. Set TITAN_DATA_BASE to the folder containing M1 CSVs.\n"
            + "\n".join(missing)
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    merged = stage_1_forex_15m()
    merged = stage_2_cross_pair(merged)
    merged = stage_3_macro(merged)
    merged = stage_4_calendar(merged)
    merged = stage_5_time(merged)
    merged = stage_5b_regimes(merged)
    merged = stage_6_targets(merged)

    merged = merged.loc[DATE_START:DATE_END]
    
    merged = merged[merged.index.dayofweek < 5]

    merged = merged.dropna(subset=["target_EURUSD_ret_12"])
    
    merged = merged.ffill().fillna(0)

    log(f"\n{'='*70}")
    log(f"           TITAN 15M DATASET — ENHANCED QUALITY REPORT")
    log(f"{'='*70}")
    log(f"  Shape            : {merged.shape[0]:,} rows × {merged.shape[1]} columns")
    log(f"  Date range       : {merged.index.min()} → {merged.index.max()}")
    
    merged.to_csv(OUT, index_label="datetime")
    file_size = OUT.stat().st_size / 1e6
    log(f"\n  💾 Saved → {OUT}")
    log(f"  📦 File size: {file_size:.1f} MB")
    log(f"{'='*70}\n")
    
    elapsed = (datetime.now() - t0).total_seconds()
    log(f"✨ Done in {elapsed:.0f}s")

if __name__ == "__main__":
    main()
