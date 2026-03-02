"""
titan_mt5_live.py  —  LIVE METATRADER 5 TRADING BOT
===================================================
TITAN-NL v5.0  |  30M Bars  |  Online Evolution  |  Active-Pair Filter

This script connects to MetaTrader 5, fetches M30 candles,
builds the feature matrix, evolves the model on newly closed
bars using RealPnLLoss, predicts continuous signals, and
executes live market orders ONLY for ACTIVE_PAIRS.

REQUIREMENTS:
  pip install MetaTrader5 yfinance pandas numpy torch scikit-learn
"""

import os, sys, json, pickle, math, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
import MetaTrader5 as mt5
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ────────────────────────────────────────────────────────────────────
MT5_LOGIN    = 103471276
MT5_PASSWORD = "Tm!z6oJh"
MT5_SERVER   = "MetaQuotes-Demo"

MAGIC_NUMBER = 7772026
RISK_PCT     = 0.02          # 2% risk per trade (was 5% — more conservative)
MAX_LOTS     = 2.0

PAIRS        = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
MT5_SYMBOLS  = {p: p for p in PAIRS}

# Pairs with positive backtest Sharpe — only these get live orders.
# Set to PAIRS to trade all, or subset to filter.
ACTIVE_PAIRS = ['USDJPY', 'AUDUSD']   # EURUSD=-3.2, GBPUSD=-9.0 disabled

SIG_THRESHOLD = 0.02   # model AvgSig ~0.042 in backtest — 0.10 was too high to ever fire

NUM_NODES       = len(PAIRS)
D_MODEL         = 96           # must match the trained model
CMS_CHUNK_SIZES = [16, 64, 256]
ONLINE_LR       = 1e-5         # fine-tune rate (not re-train)
ONLINE_EVOLVE   = False        # True = 1 gradient step per bar (fast even on CPU)
                               # False = pure inference only (recommended on slow CPU)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_HERE = os.getcwd()
MODEL_PATH  = os.path.join(_HERE, 'Best_TITAN_EVOLVING.pth')   # v5.0 30m model
SCALER_PATH = os.path.join(_HERE, 'titan_scaler.pkl')
SCHEMA_PATH = os.path.join(_HERE, 'titan_feature_schema.json')
STATE_PATH  = os.path.join(_HERE, 'titan_live_states_2026.pt')
NEW_MODEL_P = os.path.join(_HERE, 'titan_mt5_live.pth')  # evolving live weights


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (Imported from titan.py)
# ─────────────────────────────────────────────────────────────────────────────
from titan import NestedGraphTitanNL, RealPnLLoss



# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def pair_features(data: pd.DataFrame, pair: str) -> pd.DataFrame:
    if data.empty: return pd.DataFrame()
    c = data['Close']; h = data['High']; lo = data['Low']
    o = data['Open']; v = data.get('Volume', pd.Series(0, index=data.index))
    
    lr = np.log(c / c.shift(1)).fillna(0)
    f = {f'{pair}_ret': lr}
    std20 = c.rolling(20).std().fillna(1e-8)
    sma20 = c.rolling(20).mean().fillna(c)
    
    f[f'{pair}_ret_4bar']  = np.log(c / c.shift(4)).fillna(0)
    f[f'{pair}_ret_16bar'] = np.log(c / c.shift(16)).fillna(0)
    f[f'{pair}_vol_20']    = lr.rolling(20).std().fillna(0)
    f[f'{pair}_vol_60']    = lr.rolling(60).std().fillna(0)
    f[f'{pair}_atr14']     = ((h - lo).rolling(14).mean() / (c+1e-8)).fillna(0)
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-8)
    f[f'{pair}_rsi14'] = (100 - (100 / (1 + rs))).fillna(50)
    
    ll = lo.rolling(14).min(); hh = h.rolling(14).max()
    stk = 100 * (c - ll) / (hh - ll + 1e-8)
    f[f'{pair}_stoch_k'] = stk.fillna(50)
    f[f'{pair}_stoch_d'] = stk.rolling(3).mean().fillna(50)
    f[f'{pair}_williams_r'] = (-100 * (hh - c) / (hh - ll + 1e-8)).fillna(-50)
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26; sig = macd.ewm(span=9, adjust=False).mean()
    f[f'{pair}_macd']         = macd.fillna(0)
    f[f'{pair}_macd_sig']     = sig.fillna(0)
    f[f'{pair}_macd_signal']  = sig.fillna(0)
    f[f'{pair}_macd_hist']    = (macd - sig).fillna(0)
    f[f'{pair}_macd_histogram']= f[f'{pair}_macd_hist']
    
    f[f'{pair}_roc4']  = c.pct_change(4).fillna(0)
    f[f'{pair}_roc16'] = c.pct_change(16).fillna(0)
    f[f'{pair}_roc64'] = c.pct_change(64).fillna(0)
    
    bbu = sma20 + 2*std20; bbl = sma20 - 2*std20
    f[f'{pair}_bb_pctb']  = ((c - bbl) / (bbu - bbl + 1e-8)).fillna(0.5)
    f[f'{pair}_bb_width'] = ((bbu - bbl) / (sma20 + 1e-8)).fillna(0)
    
    sma50 = c.rolling(50).mean().fillna(c)
    ema20 = c.ewm(span=20, adjust=False).mean()
    f[f'{pair}_price_sma20_ratio'] = (c / (sma20 + 1e-8) - 1).fillna(0)
    f[f'{pair}_price_sma50_ratio'] = (c / (sma50 + 1e-8) - 1).fillna(0)
    f[f'{pair}_sma20'] = sma20
    f[f'{pair}_sma50'] = sma50
    f[f'{pair}_ema20'] = ema20
    
    rng = h - lo + 1e-8; body = (c - o).abs()
    f[f'{pair}_body_ratio']       = (body / rng).fillna(0)
    f[f'{pair}_upper_wick_ratio'] = ((h - c.clip(lower=o)) / rng).fillna(0)
    f[f'{pair}_lower_wick_ratio'] = ((c.clip(upper=o) - lo) / rng).fillna(0)
    f[f'{pair}_close_position']   = ((c - lo) / rng).fillna(0.5)
    f[f'{pair}_vol_ratio_20']     = (v / (v.rolling(20).mean() + 1e-8)).fillna(1)
    
    f[f'{pair}_close'] = c; f[f'{pair}_open'] = o
    f[f'{pair}_high']  = h; f[f'{pair}_low']  = lo
    f[f'{pair}_bullish'] = (c > c.shift(1)).astype(float).fillna(0)
    f[f'{pair}_cci'] = ((c - sma20) / (0.015 * std20 + 1e-8)).fillna(0)
    
    f[f'{pair}_sma5']   = c.rolling(5).mean().ffill().fillna(c)
    f[f'{pair}_sma10']  = c.rolling(10).mean().ffill().fillna(c)
    f[f'{pair}_ema12']  = ema12
    f[f'{pair}_ema26']  = ema26
    f[f'{pair}_roc1']  = c.pct_change(1).fillna(0)
    f[f'{pair}_roc5']  = c.pct_change(5).fillna(0)
    f[f'{pair}_roc10'] = c.pct_change(10).fillna(0)
    f[f'{pair}_bb_upper'] = bbu
    f[f'{pair}_bb_lower'] = bbl
    f[f'{pair}_daily_range']  = ((h - lo) / (c + 1e-8)).fillna(0)
    f[f'{pair}_daily_return'] = lr
    direction = np.sign(c.diff().fillna(0))
    obv_raw = (direction * v).cumsum()
    f[f'{pair}_obv'] = (obv_raw / (obv_raw.std() + 1e-8)).fillna(0)
    f[f'{pair}_volume'] = v
    for sess in ['tokyo', 'london', 'ny', 'overlap']:
        f[f'{pair}_session_{sess}'] = 0.0
    return pd.DataFrame(f, index=c.index)

def macro_features(macro_raw: dict, idx: pd.Index) -> pd.DataFrame:
    ALIASES = {
        'oil':         ['oil', 'wti', 'WTI', 'CRUDE_OIL', 'CL'],
        'brent':       ['brent', 'BRENT', 'BZ'],
        'yield_us2y':  ['yield_us2y', 'yield_US02Y', 'IRX'],
        'yield_us5y':  ['yield_us5y', 'yield_US05Y', 'FVX'],
        'yield10y':    ['yield10y', 'US10Y', 'TNX', 'yield_US10Y', 'macro_us_rate'],
        'yield_us30y': ['yield_us30y', 'yield_US30Y', 'TYX'],
        'gold':        ['gold', 'GOLD', 'GC', 'XAU'],
        'vix':         ['vix', 'VIX', 'CBOE_VIX'],
        'dxy':         ['dxy', 'DXY', 'usd_index_proxy', 'dollar_index', 'DX'],
        'sp500':       ['sp500', 'SP500', 'SPX', 'GSPC', 'us_equity', 'macro_sp500'],
        'natgas':      ['natgas', 'NATGAS', 'NG', 'natural_gas'],
        'silver':      ['silver', 'SILVER', 'SI', 'XAG'],
        'copper':      ['copper', 'COPPER', 'HG'],
    }
    frames = []
    closes = {}
    for name, data in macro_raw.items():
        if data is None or data.empty: continue
        cl  = data['Close'].squeeze().reindex(idx).ffill().fillna(0)
        lr  = np.log(cl / cl.shift(1)).fillna(0)
        v20 = lr.rolling(20).std().fillna(0)
        chg = cl.diff().fillna(0)
        mom = cl.pct_change(21).fillna(0)
        closes[name] = cl
        row = {}
        for alias in ALIASES.get(name, [name]):
            row[f'{alias}_close']  = cl; row[f'{alias}_ret']    = lr
            row[f'{alias}_return'] = lr; row[f'{alias}_vol_20'] = v20
            row[f'{alias}_vol']    = v20; row[f'{alias}_chg']    = chg
            row[f'{alias}_mom_chg']= mom; row[alias]             = cl
        frames.append(pd.DataFrame(row, index=idx))
    if 'gold' in closes and 'oil' in closes:
        frames.append(pd.DataFrame({'gold_oil_ratio': closes['gold'] / (closes['oil']+1e-8)}, index=idx))
    if 'gold' in closes and 'silver' in closes:
        frames.append(pd.DataFrame({'gold_silver_ratio': closes['gold'] / (closes['silver']+1e-8)}, index=idx))
    y10  = closes.get('yield10y',   pd.Series(0, index=idx))
    y2   = closes.get('yield_us2y', pd.Series(0, index=idx))
    y30  = closes.get('yield_us30y', pd.Series(0, index=idx))
    frames.append(pd.DataFrame({
        'yield_curve_US_10Y_2Y': (y10 - y2).fillna(0),
        'yield_curve_US_30Y_10Y': (y30 - y10).fillna(0),
    }, index=idx))
    return pd.concat(frames, axis=1).fillna(0) if frames else pd.DataFrame(index=idx)

def time_features(idx: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({
        'day_of_week':  [float(d.weekday()) for d in idx],
        'month':        [float(d.month)     for d in idx],
        'quarter':      [float((d.month-1)//3+1) for d in idx],
        'day_of_year':  [float(d.timetuple().tm_yday) for d in idx],
        'week_of_year': [float(d.isocalendar()[1])    for d in idx],
    }, index=idx)

def cross_pair_features(forex: dict, common_idx: pd.Index) -> pd.DataFrame:
    closes = {p: d['Close'].squeeze().reindex(common_idx).ffill().fillna(0) for p, d in forex.items()}
    lr = {p: np.log(c / c.shift(1)).fillna(0) for p, c in closes.items()}
    rows = {}
    for p1, p2 in [('EURUSD','GBPUSD'), ('EURUSD','AUDUSD'), ('EURUSD','USDJPY'),
                   ('GBPUSD','AUDUSD'), ('GBPUSD','USDJPY'), ('AUDUSD','USDJPY')]:
        if p1 in lr and p2 in lr:
            rows[f'corr_{p1}_{p2}_20'] = lr[p1].rolling(20).corr(lr[p2]).fillna(0)
    for p1, p2 in [('EURUSD','GBPUSD'), ('EURUSD','AUDUSD'), ('EURUSD','USDJPY')]:
        if p1 in closes and p2 in closes:
            rows[f'spread_{p1}_{p2}'] = (np.log(closes[p1]+1e-8) - np.log(closes[p2]+1e-8)).fillna(0)
    return pd.DataFrame(rows, index=common_idx)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (MT5 + Yahoo)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_mt5_data(num_bars: int = 500) -> dict:
    """Fetch M30 candles (was D1) to match 30m training data."""
    forex = {}
    for pair in PAIRS:
        sym = MT5_SYMBOLS.get(pair, pair)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M30, 0, num_bars)
        if rates is None or len(rates) == 0:
            print(f"[!] Failed to fetch MT5 data for {sym}: {mt5.last_error()}")
            continue
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open':'Open','high':'High','low':'Low',
                           'close':'Close','real_volume':'Volume'}, inplace=True)
        forex[pair] = df
    return forex

def fetch_macro_data(start_date, end_date) -> dict:
    MACRO_TICKERS = {
        'oil': 'CL=F', 'brent': 'BZ=F', 'yield_us2y': '^IRX', 'yield_us5y': '^FVX',
        'yield10y': '^TNX', 'yield_us30y': '^TYX', 'gold': 'GC=F', 'vix': '^VIX',
        'dxy': 'DX-Y.NYB', 'sp500': '^GSPC', 'natgas': 'NG=F', 'silver': 'SI=F', 'copper': 'HG=F'
    }
    macro = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                df.index = df.index.normalize()
                macro[name] = df
        except Exception:
            pass
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# MT5 EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def close_all_positions(pair: str):
    sym = MT5_SYMBOLS.get(pair, pair)
    positions = mt5.positions_get(symbol=sym)
    if not positions: return
    for pos in positions:
        tick = mt5.symbol_info_tick(sym)
        price = tick.ask if pos.type == mt5.ORDER_TYPE_SELL else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Titan Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[!] Close failed for {sym}: {res.retcode}")

def open_position(pair: str, signal: float, balance: float):
    sym = MT5_SYMBOLS.get(pair, pair)
    info = mt5.symbol_info(sym)
    if not info:
        print(f"[!] Symbol {sym} not found in MT5")
        return

    if abs(signal) < SIG_THRESHOLD:
        print(f"    [{pair}] Signal {signal:+.3f} too weak (< {SIG_THRESHOLD}), staying flat.")
        return
    if pair not in ACTIVE_PAIRS:
        print(f"    [{pair}] Inactive pair — signal suppressed (Sharpe < 0 in backtest).")
        return

    order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
    
    # Calculate Lot Size based on risk and signal strength
    # Example: 5% risk * signal strength
    risk_amount = balance * RISK_PCT * abs(signal)
    
    # Heuristic lot sizing (assumes $10 per pip for 1 standard lot for simplicity)
    # Ideally use ATR or stop loss distance here.
    lot_size = risk_amount / 1000.0  
    
    lot_size = max(info.volume_min, min(info.volume_max, MAX_LOTS, lot_size))
    # Round to allowed step
    lot_size = round(lot_size / info.volume_step) * info.volume_step
    
    tick = mt5.symbol_info_tick(sym)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Titan Open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[!] Open failed for {sym}: {res.retcode} - {res.comment}")
    else:
        print(f"    [{pair}] Opened {'BUY' if signal>0 else 'SELL'} {lot_size} lots at {price}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Missing artifacts:\n{MODEL_PATH}\n{SCALER_PATH}\n{SCHEMA_PATH}")
    schema = json.load(open(SCHEMA_PATH))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    
    model = NestedGraphTitanNL(num_nodes=NUM_NODES, feats_per_node=schema['feats_per_node'],
                               d_model=D_MODEL, cms_chunk_sizes=CMS_CHUNK_SIZES).to(DEVICE)
    # Always start from the base model (Best_TITAN_EVOLVING.pth).
    # titan_mt5_live.pth is only used if ONLINE_EVOLVE is True AND it was
    # saved MORE RECENTLY than the base model (i.e. by this live session).
    use_live = (
        ONLINE_EVOLVE and
        os.path.exists(NEW_MODEL_P) and
        os.path.exists(MODEL_PATH) and
        os.path.getmtime(NEW_MODEL_P) > os.path.getmtime(MODEL_PATH)
    )
    load_path = NEW_MODEL_P if use_live else MODEL_PATH
    print(f"    Loading: {os.path.basename(load_path)}")
    ckpt = torch.load(load_path, map_location=DEVICE, weights_only=True)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()

    states = None
    if os.path.exists(STATE_PATH):
        states = torch.load(STATE_PATH, map_location=DEVICE, weights_only=False)
        # Move states to current device
        if isinstance(states, dict):
            states = states.get('delta_M', [])
        if isinstance(states, list):
            states = [s.to(DEVICE) if s is not None else None for s in states]
    return model, scaler, schema, states

def p(msg):
    """Flushed print: shows immediately even on buffered/slow terminals."""
    print(msg, flush=True)

def run_daily_cycle():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    p(f"\n{'='*55}")
    p(f"[{now}]  New M30 bar — starting cycle")
    p(f"{'='*55}")
    p("  [1/5] Loading model artifacts...")
    model, scaler, schema, states = load_artifacts()
    feats_per_node = schema['feats_per_node']
    p(f"        feats/node={feats_per_node}  device={DEVICE}")

    p("  [2/5] Fetching M30 bars from MT5...")
    forex = fetch_mt5_data(num_bars=500)
    if not forex:
        p("  [!] No forex data. Aborting cycle.")
        return
    idx = forex[PAIRS[0]].index
    p(f"        {len(idx)} bars  ({idx[0]}  to  {idx[-1]})")

    p("  [3/5] Fetching macro (yfinance)...")
    start_str = idx[0].strftime('%Y-%m-%d')
    end_str   = (idx[-1] + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    macro = fetch_macro_data(start_str, end_str)
    p(f"        {len(macro)} macro series ok")

    p("  [4/5] Building & scaling features (direct schema match)...")

    # Map macro ticker name -> expected column name in schema
    # Schema expects: oil_close, gold_close, copper_close, sp500_close,
    #                 dxy_close, vix_close, yield10y_close, yield2y_close, nikkei_close
    MACRO_COL_MAP = {
        'oil':      'oil_close',
        'gold':     'gold_close',
        'copper':   'copper_close',
        'sp500':    'sp500_close',
        'dxy':      'dxy_close',
        'vix':      'vix_close',
        'yield10y': 'yield10y_close',
        'yield_us2y': 'yield2y_close',
        'nikkei':   'nikkei_close',
    }

    # Build macro close series aligned to idx
    macro_closes = {}
    for mname, col_name in MACRO_COL_MAP.items():
        if mname in macro and not macro[mname].empty:
            raw_cl = macro[mname]['Close'].squeeze()
            if raw_cl.index.tz is not None:
                raw_cl.index = raw_cl.index.tz_localize(None)
            s = raw_cl.reindex(idx, method='ffill').ffill().bfill().fillna(0.0)
        else:
            s = pd.Series(0.0, index=idx)
        macro_closes[col_name] = s.values.astype(np.float32)

    node_arrays = []
    matched_counts = []
    for pair in PAIRS:
        expected = schema['node_cols'].get(pair, [])  # e.g. ['EURUSD_Open',...,'oil_close',...]
        mat = np.zeros((len(idx), feats_per_node), dtype=np.float32)
        matched = 0
        pair_df = forex.get(pair, pd.DataFrame(index=idx))
        pair_df_ri = pair_df.reindex(idx).ffill().fillna(0.0)
        for ci, col in enumerate(expected[:feats_per_node]):
            if col == f'{pair}_Open'  and 'Open'  in pair_df_ri.columns:
                mat[:, ci] = pair_df_ri['Open'].values.astype(np.float32);  matched += 1
            elif col == f'{pair}_High'  and 'High'  in pair_df_ri.columns:
                mat[:, ci] = pair_df_ri['High'].values.astype(np.float32);  matched += 1
            elif col == f'{pair}_Low'   and 'Low'   in pair_df_ri.columns:
                mat[:, ci] = pair_df_ri['Low'].values.astype(np.float32);   matched += 1
            elif col == f'{pair}_Close' and 'Close' in pair_df_ri.columns:
                mat[:, ci] = pair_df_ri['Close'].values.astype(np.float32); matched += 1
            elif col in macro_closes:
                mat[:, ci] = macro_closes[col]; matched += 1
        node_arrays.append(mat)
        matched_counts.append(matched)

    total_matched = sum(matched_counts) // max(len(PAIRS), 1)
    p(f"        matched {total_matched}/{feats_per_node} features per node  {matched_counts}")

    master = np.stack(node_arrays, axis=1) # [T, N, F]
    
    # Scale + robust-clip every feature channel
    scaled = np.nan_to_num(scaler.transform(master.reshape(-1, feats_per_node))
                           .reshape(len(idx), NUM_NODES, feats_per_node), nan=0.0)
    for ci in range(feats_per_node):
        col = scaled[:, :, ci]
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med))) + 1e-8
        scaled[:, :, ci] = np.clip((col - med) / mad, -5.0, 5.0)

    
    if ONLINE_EVOLVE and len(idx) >= 3:
        p("  [online-evolve] 1 gradient step...")
        x_train = torch.FloatTensor(scaled[-3:-2]).unsqueeze(1).to(DEVICE)  # [B=1, S=1, N=4, F=174]
        
        # Calculate actual returns from T-2 close to T-1 close
        actual_returns = np.zeros((1, NUM_NODES), dtype=np.float32)
        for p_idx, pair in enumerate(PAIRS):
            c = forex[pair]['Close'].values
            actual_returns[0, p_idx] = np.log((c[-2] + 1e-8) / (c[-3] + 1e-8))
            
        r_train = torch.FloatTensor(actual_returns).unsqueeze(1).to(DEVICE)  # [1, 1, N]

        model.train()
        criterion = RealPnLLoss()    # v5.0: cost-aware PnL loss
        optimizer = optim.AdamW(model.parameters(), lr=ONLINE_LR, weight_decay=1e-4)
        
        optimizer.zero_grad()
        train_states = None
        if states is not None:
            if isinstance(states, dict):
                states = states.get('delta_M', [])
            train_states = [m.detach().clone() if m is not None else None for m in states]
            
        signal, next_states = model(x_train, train_states)
        signal = signal.squeeze(-1)  # [B, N]
        prev_sig = None
        loss = criterion(signal, r_train, prev_sig=prev_sig)
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.10)
            optimizer.step()
        
        states = next_states
        p(f"        loss={loss.item():.5f}  weights saved")
        torch.save(model.state_dict(), NEW_MODEL_P)
        torch.save([s.detach().cpu() if s is not None else None for s in states], STATE_PATH)


    p("  [5/5] Running inference...")
    model.eval()
    with torch.no_grad():
        x_infer = torch.FloatTensor(scaled[-2:-1]).unsqueeze(1).to(DEVICE)
        signal, states = model(x_infer, states)
        signal_np = signal.squeeze(-1)[0].cpu().numpy()

    p(f"\n  SIGNALS  (threshold={SIG_THRESHOLD})")
    for i, pair in enumerate(PAIRS):
        tag = "ACTIVE" if pair in ACTIVE_PAIRS else "inactive"
        p(f"    {pair:8s}  sig={signal_np[i]:+.4f}  [{tag}]")

    p(f"\n  EXECUTION")
    acc_info = mt5.account_info()
    if acc_info is None:
        p("  [!] MT5 account info unavailable — orders skipped.")
        return
    balance = acc_info.balance
    p(f"    Balance: {balance:.2f}")
    for i, pair in enumerate(PAIRS):
        sig = float(signal_np[i])
        # Zero signal for inactive pairs — they keep any existing flat position
        if pair not in ACTIVE_PAIRS:
            sig = 0.0
        close_all_positions(pair)
        if abs(sig) >= SIG_THRESHOLD:
            open_position(pair, sig, balance)
        else:
            p(f"    [{pair}] Flat (|sig|={abs(sig):.3f} < {SIG_THRESHOLD})")

    p(f"  Cycle complete. Next bar in ~30m.\n")


if __name__ == '__main__':
    print("="*60)
    print("TITAN-NL LIVE MT5 TRADING BOT STARTING...")
    print("="*60)
    
    if not mt5.initialize():
        print(f"[!] MT5 initialize() failed, error code = {mt5.last_error()}")
        sys.exit()

    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"[!] MT5 login() failed, error code = {mt5.last_error()}")
        sys.exit()

        
    print(f"Successfully connected to MT5 Account: {mt5.account_info().login}")
    
    # For a daily strategy, we just need to wake up once a day after candle close.
    # Assuming GMT+2 broker (standard for forex), D1 candle closes at 00:00 server time.
    # We poll every hour, and trigger when the D1 timestamp changes.
    
    last_processed_bar = None
    
    run_daily_cycle()  # initial run on start
    
    try:
        while True:
            # Poll latest M30 bar time
            rates = mt5.copy_rates_from_pos(MT5_SYMBOLS[PAIRS[0]], mt5.TIMEFRAME_M30, 0, 1)
            if rates is not None and len(rates) > 0:
                current_bar = rates[0]['time']
                if last_processed_bar is None:
                    last_processed_bar = current_bar
                elif current_bar > last_processed_bar:
                    print("\nNew 30m bar detected. Processing...")
                    time.sleep(10)  # brief settle time after new bar
                    run_daily_cycle()
                    last_processed_bar = current_bar
            time.sleep(60)  # check every minute
            
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        mt5.shutdown()
