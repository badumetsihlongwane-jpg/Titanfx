"""
titan_live_2026.py  —  STANDALONE VERSION
==========================================
TITAN-NL v5.0  Live 2026 Calibration + Backtest
NO external titan.py import — all model classes are embedded here.

REQUIRES (in /kaggle/working after running titan.py):
  Best_TITAN_EVOLVING.pth        ← model weights (v5.0 dual-head)
  titan_scaler.pkl               ← RobustScaler trained on 2010-2023
  titan_feature_schema.json      ← ordered column names
  titan_final_memory_state.pt    ← optional: last memory state

USAGE (Kaggle):
  !pip install yfinance -q
  %run titan_live_2026.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json, pickle, math, warnings, io
from datetime import datetime
from typing import List, Optional, Tuple
warnings.filterwarnings('ignore')

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import RobustScaler

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Run:  !pip install yfinance -q  then re-run this cell")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PAIRS           = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
NUM_NODES       = len(PAIRS)
D_MODEL         = 96         # Must match titan.py D_MODEL used during training
CMS_CHUNK_SIZES = [16, 64, 256]
CALIB_BARS      = 600      # bars used for memory warmup + optional calibration (~5 days)
RUN_CALIBRATION = True     # True = update weights; False = warmup only, pure inference
ONLINE_LR       = 1e-5    # Fine-tune rate (3e-4 was too high — caused weight drift over 2000 bars)
BARSPERYEAR     = 22176    # 15m bars per year

# ── Real PnL Loss hyperparams (must match titan.py) ──────────────────
SPREAD_BPS    = 1.0
LAMBDA_TC     = 0.5
LAMBDA_CVAR   = 0.1
TARGET_VOL    = 0.001
CVAR_QUANTILE = 0.10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# File finder — searches /kaggle/working and cwd
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

def _find(name: str) -> str:
    for d in ('/kaggle/working', _HERE, '.'):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"'{name}' not found. Run titan.py training first to generate it.\n"
        f"Searched: /kaggle/working, {_HERE}")

FOREX_TICKERS = {p: f"{p}=X".replace("USDJPY=X", "JPY=X") for p in PAIRS}
FOREX_TICKERS['USDJPY'] = 'USDJPY=X'   # yfinance uses full pair for JPY

MACRO_TICKERS = {
    'oil':      'CL=F',      # WTI Crude
    'yield10y': '^TNX',      # US 10-Year Yield (x10 = %)
    'yield2y':  '^IRX',      # US 2-Year proxy (13-week bill)
    'gold':     'GC=F',
    'vix':      '^VIX',
    'dxy':      'DX-Y.NYB',
    'sp500':    '^GSPC',
    'natgas':   'NG=F',
    'silver':   'SI=F',
    'copper':   'HG=F',
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (embedded — no titan.py import needed)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: All class definitions here must stay in sync with titan.py v5.0
class SelfModifyingDeltaMemory(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.proj_q  = nn.Linear(d_model, d_model, bias=False)
        self.proj_k  = nn.Linear(d_model, d_model, bias=False)
        self.proj_v  = nn.Linear(d_model, d_model, bias=False)
        self.value_generator = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.eta_proj   = nn.Sequential(
            nn.Linear(d_model, d_model//4), nn.SiLU(), nn.Linear(d_model//4, 1), nn.Sigmoid())
        self.alpha_proj = nn.Sequential(
            nn.Linear(d_model, d_model//4), nn.SiLU(), nn.Linear(d_model//4, 1), nn.Sigmoid())
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer('init_memory', torch.zeros(d_model, d_model))

    def forward(
        self,
        x: torch.Tensor,
        prev_M: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        b, s, n, f = x.shape
        x_flat = x.view(b * n, s, f)
        residual = x_flat

        q     = self.proj_q(x_flat)
        k     = self.proj_k(x_flat)
        v     = self.proj_v(x_flat)
        v_hat = self.value_generator(v)
        eta   = self.eta_proj(x_flat)   * 0.1 + 0.01
        alpha = self.alpha_proj(x_flat) * 0.5 + 0.5

        if prev_M is not None:
            M = prev_M
        else:
            M = self.init_memory.unsqueeze(0).expand(b * n, -1, -1).clone()

        outputs = []
        for t in range(s):
            q_t      = q[:, t, :]
            k_t_norm = F.normalize(k[:, t, :], dim=-1)
            v_t      = v_hat[:, t, :]
            eta_t    = eta[:, t, :].unsqueeze(-1)
            alpha_t  = alpha[:, t, :].unsqueeze(-1)

            out_t = torch.bmm(M, q_t.unsqueeze(-1)).squeeze(-1)

            Mk = torch.bmm(M, k_t_norm.unsqueeze(-1))
            M  = (alpha_t * M
                  - eta_t * torch.bmm(Mk, k_t_norm.unsqueeze(-2))
                  + eta_t * torch.bmm(v_t.unsqueeze(-1), k_t_norm.unsqueeze(-2)))
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        output = self.norm(self.dropout(self.out_proj(output)) + residual)
        return output.view(b, s, n, f), M


class ContinuumMemoryMLP(nn.Module):
    def __init__(self, d_model: int, chunk_sizes: List[int] = [16, 64, 256],
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.chunk_sizes = chunk_sizes
        self.num_levels  = len(chunk_sizes)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * expansion),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * expansion, d_model),
                nn.Dropout(dropout)
            ) for _ in range(self.num_levels)
        ])
        self.level_weights = nn.Parameter(torch.ones(self.num_levels))
        self.level_norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_levels)])
        self.final_norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        b, s, n, f = x.shape
        x_flat = x.view(b * s * n, f)
        level_outputs = []
        for level_idx, (mlp, norm) in enumerate(zip(self.mlps, self.level_norms)):
            out = mlp(x_flat)
            if self.training:
                drop_p = [0.3, 0.15, 0.0][level_idx]
                out = F.dropout(out, p=drop_p, training=True)
            level_outputs.append(norm(out + x_flat))
        weights    = F.softmax(self.level_weights, dim=0)
        aggregated = sum(w * o for w, o in zip(weights, level_outputs))
        return self.final_norm(aggregated).view(b, s, n, f)


class MarketRegimeMemory(nn.Module):
    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model   = d_model
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(),
            nn.Linear(d_model, 3), nn.Softmax(dim=-1)
        )
        self.regime_eta   = nn.Parameter(torch.tensor([0.1, 0.05, 0.2]))
        self.regime_alpha = nn.Parameter(torch.tensor([0.8, 0.9, 0.6]))
        self.q_graph      = nn.Linear(d_model, d_model)
        self.k_graph      = nn.Linear(d_model, d_model)
        self.v_graph      = nn.Linear(d_model, d_model)
        self.gate_net     = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 3),
            nn.Linear(d_model * 3 + 3, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, 1.5)
        self.gate_act = nn.Sigmoid()
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        state    = x[:, -3:, :, :].mean(dim=1)
        b, n, d  = state.shape
        residual = state

        global_mean = state.mean(dim=1, keepdim=True)
        global_std  = state.std(dim=1, keepdim=True)

        regime_input = torch.cat([state, global_mean.expand(-1, n, -1)], dim=-1)
        regime_probs = self.regime_detector(regime_input)

        gate_input   = torch.cat([state, global_mean.expand(-1, n, -1),
                                  global_std.expand(-1, n, -1), regime_probs], dim=-1)
        alpha        = self.gate_act(self.gate_net(gate_input))

        Q = self.q_graph(state)
        K = self.k_graph(state)
        V = self.v_graph(state)

        attn_scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(attn_scores, dim=-1)

        I            = torch.eye(n, device=x.device).unsqueeze(0).expand(b, -1, -1)
        mixed_weights = (alpha * I) + ((1 - alpha) * attn_weights)

        out = self.norm(self.dropout(torch.matmul(mixed_weights, V)) + residual)
        return out, alpha, attn_weights, regime_probs


class NestedGraphTitanNL(nn.Module):
    """
    v5.0: Dual-head — direction [-1,1] × gate [0,1].
    Effective signal = direction * gate  (gate ≈ 0 means flat).
    """
    def __init__(self, num_nodes: int = NUM_NODES, feats_per_node: int = 73,
                 d_model: int = D_MODEL, num_layers: int = 2, dropout: float = 0.3,
                 cms_chunk_sizes: List[int] = CMS_CHUNK_SIZES):
        super().__init__()
        mid_dim = min(d_model, max(64, feats_per_node // 3))
        self.embedding  = nn.Sequential(nn.Linear(feats_per_node, mid_dim), nn.SiLU(),
                                        nn.Linear(mid_dim, d_model))
        self.input_norm = nn.LayerNorm(d_model)

        max_len  = 512
        pos_emb  = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_emb = nn.Parameter(pos_emb)

        self.temporal_layers = nn.ModuleList([
            SelfModifyingDeltaMemory(d_model, dropout) for _ in range(num_layers)
        ])
        self.cms           = ContinuumMemoryMLP(d_model, cms_chunk_sizes, expansion=3, dropout=dropout)
        self.regime_memory = MarketRegimeMemory(num_nodes, d_model, dropout)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout)
        )
        # Head 1 — direction [-1, 1]
        self.direction_head = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Tanh())
        # Head 2 — gate [0, 1]
        self.gate_head = nn.Sequential(nn.Linear(d_model // 2, 1), nn.Sigmoid())
        nn.init.constant_(self.gate_head[0].bias, 0.5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.gate_head[0].bias, 0.5)

    def forward(
        self,
        x: torch.Tensor,
        prev_states=None,
        return_attn: bool = False,
        step: int = 0
    ):
        b, s, n, f = x.shape
        x   = self.embedding(x)
        pos = self.pos_emb[:, :s, :].unsqueeze(2).expand(b, s, n, -1)
        x   = self.input_norm(x + pos)

        current_states = []
        for i, layer in enumerate(self.temporal_layers):
            p_M      = prev_states[i] if prev_states is not None else None
            x, new_M = layer(x, p_M)
            current_states.append(new_M)

        x = self.cms(x, step=step)
        graph_out, alpha, attn_weights, regime_probs = self.regime_memory(x)

        trunk     = self.trunk(graph_out)            # [B, N, D//2]
        direction = self.direction_head(trunk)       # [B, N, 1]
        gate      = self.gate_head(trunk)            # [B, N, 1]
        signal    = direction * gate                 # effective position

        if return_attn:
            return signal, current_states, attn_weights, alpha, gate
        return signal, current_states


class RealPnLLoss(nn.Module):
    """Cost-aware PnL loss — matches titan.py v5.0 training objective."""
    def __init__(
        self,
        spread_bps:    float = SPREAD_BPS,
        lambda_tc:     float = LAMBDA_TC,
        lambda_cvar:   float = LAMBDA_CVAR,
        target_vol:    float = TARGET_VOL,
        cvar_quantile: float = CVAR_QUANTILE,
        lambda_l2:     float = 0.005,   # reduced: lets model take strong short positions
    ):
        super().__init__()
        self.spread      = spread_bps * 1e-4
        self.lambda_tc   = lambda_tc
        self.lambda_cvar = lambda_cvar
        self.target_vol  = target_vol
        self.quantile    = cvar_quantile
        self.lambda_l2   = lambda_l2

    def forward(
        self,
        signal:   torch.Tensor,
        targets:  torch.Tensor,
        prev_sig: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sig          = signal.squeeze(-1)                         # [B, N]
        realized_vol = targets.std(dim=1).clamp(min=1e-8)        # [B, N]
        scale        = (self.target_vol / realized_vol).clamp(0.1, 3.0)
        pos          = sig * scale
        r_net        = targets.sum(dim=1)
        pnl          = pos * r_net

        if prev_sig is not None:
            turnover = (sig - prev_sig).abs().mean()
        else:
            turnover = sig.abs().mean()
        tc_cost = self.lambda_tc * turnover * self.spread

        pos_expanded = pos.unsqueeze(1).expand_as(targets)
        bar_pnl      = (pos_expanded * targets).view(-1)
        k            = max(1, int(self.quantile * bar_pnl.numel()))
        worst_k      = torch.topk(bar_pnl, k, largest=False).values
        cvar_pen     = self.lambda_cvar * (-worst_k.mean())

        l2_pen = self.lambda_l2 * (sig ** 2).mean()
        return -pnl.mean() + tc_cost + cvar_pen + l2_pen


# Keep old name as alias for backward compatibility with saved .pth files
ProfitMaximizationLoss = RealPnLLoss


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features(forex: dict, macro: dict, common_idx: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame(index=common_idx).fillna(0.0)

    def pct(series, periods=1): return series.pct_change(periods).fillna(0)
    def sma(series, periods): return series.rolling(periods, min_periods=1).mean()
    def ema(series, periods): return series.ewm(span=periods, adjust=False).mean()
    def rsi(series, periods):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(periods, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(periods, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs)).fillna(50)
    def atr(h, l, c, periods=14):
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=periods, adjust=False).mean()

    # 1. Base FOREX features (21 per pair)
    for p in PAIRS:
        if p not in forex: continue
        c = forex[p]['Close'].squeeze(); h = forex[p]['High'].squeeze()
        l = forex[p]['Low'].squeeze();   o = forex[p]['Open'].squeeze()
        v = forex[p].get('Volume', pd.Series(0, index=c.index)).squeeze()

        df[f"{p}_Open"] = o; df[f"{p}_High"] = h; df[f"{p}_Low"] = l
        df[f"{p}_Close"] = c; df[f"{p}_Volume"] = v

        df[f"{p}_ret_1"]  = pct(c, 1);  df[f"{p}_ret_2"]  = pct(c, 2)
        df[f"{p}_ret_4"]  = pct(c, 4);  df[f"{p}_ret_8"]  = pct(c, 8)
        df[f"{p}_ret_12"] = pct(c, 12)

        df[f"{p}_range"]      = h - l
        df[f"{p}_body"]       = (c - o).abs()
        df[f"{p}_upper_wick"] = h - pd.concat([c, o], axis=1).max(axis=1)
        df[f"{p}_lower_wick"] = pd.concat([c, o], axis=1).min(axis=1) - l
        df[f"{p}_atr_14"]     = atr(h, l, c, 14).fillna(0)

        df[f"{p}_vol_std_20"]  = pct(c).rolling(20, min_periods=1).std().fillna(0)
        df[f"{p}_sma_20_slope"] = sma(c, 20).diff() / (sma(c, 20).shift(1) + 1e-8)
        df[f"{p}_ema_cross"]   = (ema(c, 12) - ema(c, 26)) / (c + 1e-8)
        df[f"{p}_rsi_14"]      = rsi(c, 14)
        df[f"{p}_roc_4"]       = c.diff(4) / (c.shift(4) + 1e-8)

    # 2. Cross-Pair (8 cols)
    df["rs_EURUSD_GBPUSD"] = df.get("EURUSD_ret_1", 0) - df.get("GBPUSD_ret_1", 0)
    df["rs_EURUSD_AUDUSD"] = df.get("EURUSD_ret_1", 0) - df.get("AUDUSD_ret_1", 0)
    df["rs_GBPUSD_AUDUSD"] = df.get("GBPUSD_ret_1", 0) - df.get("AUDUSD_ret_1", 0)
    for p1, p2 in [("EURUSD", "GBPUSD"), ("EURUSD", "AUDUSD"), ("EURUSD", "USDJPY")]:
        if f"{p1}_ret_1" in df and f"{p2}_ret_1" in df:
            df[f"corr_{p1}_{p2}"] = df[f"{p1}_ret_1"].rolling(12, min_periods=4).corr(df[f"{p2}_ret_1"]).fillna(0)
    usd_f = (-df.get("EURUSD_ret_1", 0) - df.get("GBPUSD_ret_1", 0) - df.get("AUDUSD_ret_1", 0) + df.get("USDJPY_ret_1", 0)) / 4.0
    df["usd_factor"] = usd_f
    for p, inv in [("EURUSD", -1), ("GBPUSD", -1), ("AUDUSD", -1), ("USDJPY", 1)]:
        df[f"{p}_usd_divergence"] = (df.get(f"{p}_ret_1", 0) * inv) - usd_f

    # 3. Time features (4 cols)
    hours = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    dows = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dows / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 5.0)

    # 4. Trading sessions (5 cols) — UTC-based 15M bars
    h_utc = df.index.hour
    df["session_tokyo"]   = ((h_utc >= 0)  & (h_utc < 9)).astype(np.float32)
    df["session_london"]  = ((h_utc >= 8)  & (h_utc < 17)).astype(np.float32)
    df["session_ny"]      = ((h_utc >= 13) & (h_utc < 22)).astype(np.float32)
    df["session_overlap"] = ((h_utc >= 13) & (h_utc < 17)).astype(np.float32)  # London+NY
    df["rollover_window"] = ((h_utc == 21) | (h_utc == 22)).astype(np.float32)  # 21-23 UTC

    # 5. Macro: yields + rates (10 cols)
    def _macro_series(key):
        """Return a 15M-indexed Series for a macro key, forward-filled."""
        if macro.get(key) is None:
            return pd.Series(0.0, index=common_idx)
        s = macro[key]['Close'].squeeze().reindex(common_idx).ffill().fillna(0)
        return s

    y10  = _macro_series('yield10y') / 10.0   # ^TNX is in tenths of a percent
    y2   = _macro_series('yield2y')  / 10.0   # ^IRX
    slope_2s10s = (y10 - y2).fillna(0)

    df["yield_US02"]             = y2
    df["yield_US10"]             = y10
    df["yield_US2s10s_slope"]    = slope_2s10s
    df["yield_US02_chg"]         = y2.diff().fillna(0)
    df["yield_US10_chg"]         = y10.diff().fillna(0)
    df["yield_US2s10s_slope_chg"] = slope_2s10s.diff().fillna(0)

    # Approximate central bank rates (static values updated quarterly — good enough)
    # Source: early 2026 approximations
    _cb_rates = {'Fed': 4.33, 'ECB': 2.50, 'BoE': 4.75, 'BoJ': 0.50, 'RBA': 4.10}
    df["rate_diff_Fed_ECB"] = _cb_rates['Fed'] - _cb_rates['ECB']  # scalar
    df["rate_diff_Fed_BoE"] = _cb_rates['Fed'] - _cb_rates['BoE']
    df["rate_diff_Fed_BoJ"] = _cb_rates['Fed'] - _cb_rates['BoJ']
    df["rate_diff_Fed_RBA"] = _cb_rates['Fed'] - _cb_rates['RBA']

    # 6. Central bank balance sheet QoQ changes — set to 0 live (no real-time feed)
    for cb in ['Fed', 'ECB', 'BoE', 'BoJ', 'PBoC', 'RBA', 'BoC', 'SNB']:
        df[f"cb_{cb}_qoq_chg"] = 0.0

    # 7. Economic event features — set neutral live (no real-time event calendar)
    df["mins_to_high_event"]   = 9999.0   # large = no event imminent
    df["mins_since_high_event"] = 9999.0
    df["impact_level"]          = 0.0
    df["surprise_zscore"]       = 0.0
    for ev in ['CPI', 'NFP', 'FOMC', 'GDP', 'PMI', 'Retail_Sales', 'Unemployment',
               'Interest_Rate', 'Employment', 'Manufacturing']:
        df[f"event_{ev}"] = 0.0
    df["recent_surprise"] = 0.0

    # 8. Macro return proxies (3 cols)
    if macro.get('gold') is not None:
        gold_close = macro['gold']['Close'].squeeze().reindex(common_idx).ffill().fillna(0)
        df["gold_ret"] = gold_close.pct_change().fillna(0)
    if macro.get('oil') is not None:
        oil_close = macro['oil']['Close'].squeeze().reindex(common_idx).ffill().fillna(0)
        df["wti_ret"] = oil_close.pct_change().fillna(0)
    if macro.get('sp500') is not None:
        sp_close = macro['sp500']['Close'].squeeze().reindex(common_idx).ffill().fillna(0)
        df["sp500_ret"] = sp_close.pct_change().fillna(0)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    print(f"\n>>> Downloading 28 days of 15m intraday data for backtest")
    forex = {}
    for pair, ticker in FOREX_TICKERS.items():
        try:
            raw = yf.download(ticker, period="28d", interval="15m", progress=False, auto_adjust=True)
            if not raw.empty:
                # Localize/strip timezone so all indices are tz-naive
                if raw.index.tz is not None:
                    raw.index = raw.index.tz_convert('UTC').tz_localize(None)
                forex[pair] = raw
                print(f"    [{pair}] {len(raw)} bars")
        except Exception as e:
            print(f"    [{pair}] error: {e}")
    if not forex:
        raise RuntimeError("No forex data — check internet connection")

    # ── Build common 15M timestamp index (do NOT normalize to dates) ───────────
    # normalize() collapses all intraday bars to midnight → 14 rows, not ~1260.
    # Instead intersect on the actual datetime index.
    common = None
    for df_pair in forex.values():
        idx = df_pair.index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    print(f"    Common 15M bars: {len(common)}  ({common[0]} → {common[-1]})")

    # ── Macro: download daily, forward-fill onto the 15M grid ───────────────
    macro = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, period="14d", interval="1d", progress=False, auto_adjust=True)
            if not raw.empty:
                if raw.index.tz is not None:
                    raw.index = raw.index.tz_convert('UTC').tz_localize(None)
                # Reindex onto the 15M grid with forward-fill
                raw_reindexed = raw.reindex(common, method='ffill')
                macro[name] = raw_reindexed
            else:
                macro[name] = None
        except Exception:
            macro[name] = None
    print(f"    Macro: {sum(v is not None for v in macro.values())}/{len(MACRO_TICKERS)} ok")
    return forex, macro, common


# ─────────────────────────────────────────────────────────────────────────────
# SHARPE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def sharpe(s: np.ndarray, r: np.ndarray) -> float:
    pnl = s * r; mu = pnl.mean(); sd = pnl.std()
    return 0.0 if sd < 1e-8 else (mu / sd) * math.sqrt(BARSPERYEAR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TITAN-NL v5.0  —  Live 2026 Calibration + Backtest")
    print("=" * 60)

    # ── Load schema + scaler ──────────────────────────────────────────────
    schema: dict = json.load(open(_find('titan_feature_schema.json')))
    feats_per_node: int = schema['feats_per_node']
    scaler: RobustScaler = pickle.load(open(_find('titan_scaler.pkl'), 'rb'))
    print(f"Schema loaded: {feats_per_node} features/node")

    # ── Download + build feature matrix ──────────────────────────────────
    forex, macro, common_idx = download_data()
    T = len(common_idx)
    print(f"\n>>> Building new 73-feature matrix ({T} bars)...")

    # ── Reindex each forex pair to the common 15M index ──────────────────────
    for pair in list(forex.keys()):
        forex[pair] = forex[pair].reindex(common_idx).ffill()
    master_df = build_features(forex, macro, common_idx)

    node_arrays = []
    for pair in PAIRS:
        expected = schema['node_cols'].get(pair, [])
        # ── KEY FIX: initialize to scaler center (training median) not 0.0 ─────────────
        # RobustScaler does: (x - center_) / scale_
        # If we fill unmatched cols with 0, they become (0-median)/IQR = large negative.
        # If we fill with center_, they become (center_-center_)/IQR = 0. ← neutral.
        mat = np.tile(scaler.center_.astype(np.float32), (T, 1))  # [T, F]
        matched = 0
        for ci, col in enumerate(expected[:feats_per_node]):
            if col in master_df.columns:
                mat[:, ci] = master_df[col].values.astype(np.float32)
                matched += 1
        print(f"    [{pair}] {matched}/{feats_per_node} features matched")
        node_arrays.append(mat)

    master = np.stack(node_arrays, axis=1)   # [T, N, F]
    scaled = np.clip(
        np.nan_to_num(
            scaler.transform(master.reshape(-1, feats_per_node)).reshape(T, NUM_NODES, feats_per_node),
            nan=0.0, posinf=3.0, neginf=-3.0),
        -5.0, 5.0                            # hard-clip to match training's clamp(-10,10) spirit
    ).astype(np.float32)
    print(f"    Scaled — mean {scaled.mean():.3f}, std {scaled.std():.3f}  "
          f"(target: mean≈0.0, std≈1.0)")

    # ── Future returns (EUR log-ret shifted -1) ────────────────────────────
    # -- Triple-barrier returns per pair (matches training objective) ---
    # Training used 12-bar TP/SL/expire horizon; 1-bar log-ret was wrong.
    def _tb_rets(close_series, k_tp=2.0, k_sl=1.5, max_hold=12, atr_p=14):
        import pandas as _pd
        c   = close_series.ffill().values.astype('float64')
        Tl  = len(c)
        ret = np.diff(c, prepend=c[0])
        atr = _pd.Series(np.abs(ret)).ewm(span=atr_p, adjust=False).mean().values
        atr = np.maximum(atr, 1e-8)
        out = np.zeros(Tl, dtype='float32')
        for t in range(Tl - 1):
            entry = c[t]; tp = entry + k_tp*atr[t]; sl = entry - k_sl*atr[t]
            end_t = min(t + max_hold, Tl - 1)
            outcome = (c[end_t] - entry) / (entry + 1e-12)
            for j in range(t + 1, end_t + 1):
                if c[j] >= tp:  outcome =  k_tp*atr[t]/(entry+1e-12); break
                if c[j] <= sl:  outcome = -k_sl*atr[t]/(entry+1e-12); break
            out[t] = float(outcome)
        return out

    fut_r_list = []
    for pair in PAIRS:
        if pair in forex:
            c_ser = forex[pair]['Close'].squeeze().reindex(common_idx).ffill()
            tb = _tb_rets(c_ser)
            pct_tp = float((tb > 1e-5).mean()) * 100
            pct_sl = float((tb < -1e-5).mean()) * 100
            print(f'    [{pair}] triple-barrier: std={tb.std():.5f}  TP%={pct_tp:.1f}  SL%={pct_sl:.1f}')
        else:
            tb = np.zeros(T, dtype='float32')
        fut_r_list.append(tb)
    fut_r = np.stack(fut_r_list, axis=1)   # [T, N]  per-pair triple-barrier outcomes
    fut_r[-12:] = 0.0                       # last max_hold bars: incomplete barriers

    # ── Train / backtest split ─────────────────────────────────────────────
    c_end = min(CALIB_BARS, max(1, T - 100)) # Ensure at least 100 bars for backtest
    calib_X = torch.FloatTensor(scaled[:c_end])
    calib_R = torch.FloatTensor(fut_r[:c_end])
    bt_X    = torch.FloatTensor(scaled[c_end:])
    bt_R    = torch.FloatTensor(fut_r[c_end:])

    d0 = common_idx[0].strftime('%Y-%m-%d %H:%M'); d1 = common_idx[c_end - 1].strftime('%Y-%m-%d %H:%M')
    d2 = common_idx[c_end].strftime('%Y-%m-%d %H:%M') if c_end < T else 'N/A'; d3 = common_idx[-1].strftime('%Y-%m-%d %H:%M')
    print(f"\n  Calibration : {c_end} bars  ({d0} → {d1})")
    print(f"  Backtest    : {len(bt_X)} bars  ({d2} → {d3})")

    # ── Load model ────────────────────────────────────────────────────────
    model = NestedGraphTitanNL(
        num_nodes=NUM_NODES, feats_per_node=feats_per_node,
        d_model=D_MODEL, num_layers=2, dropout=0.3,
        cms_chunk_sizes=CMS_CHUNK_SIZES).to(DEVICE)

    model.load_state_dict(torch.load(_find('Best_TITAN_EVOLVING.pth'),
                                     map_location=DEVICE, weights_only=True))
    print(f"\nLoaded model  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Load or init memory state ─────────────────────────────────────────
    # Always start fresh: saved M-matrices carry 2010-2024 statistics;
    # passing them to 2026 live data causes activation mismatch -> NaN.
    # The 300-bar calibration loop warms the memory on live data organically.
    states = None
    mode_str = "calibration + warmup" if RUN_CALIBRATION else "memory warmup only (pure inference mode)"
    print(f"Starting with fresh memory state ({mode_str})")

    # ── CALIBRATION — 30 days of live weight updates ──────────────────────
    print(f"\n{'='*60}")
    print(f"CALIBRATION  (WARMUP_BARS={50} no-grad | calibration={'ON' if RUN_CALIBRATION else 'OFF'})")
    print(f"{'='*60}")
    criterion = RealPnLLoss()
    opt       = optim.AdamW(model.parameters(), lr=ONLINE_LR, weight_decay=1e-4)

    # Phase A: no-grad memory warmup — always runs regardless of RUN_CALIBRATION
    # Prime the M-matrices on live data before any inference or weight updates.
    WARMUP_BARS = min(50, c_end)
    model.eval()
    print(f"  Warming up memory ({WARMUP_BARS} bars, no gradient) ...")
    with torch.no_grad():
        for i in range(WARMUP_BARS):
            x_b = torch.clamp(
                calib_X[i].unsqueeze(0).unsqueeze(0).to(DEVICE), -5.0, 5.0)
            _, states = model(x_b, prev_states=states)
            states = [s.detach() for s in states]
            if any(torch.isnan(s).any() for s in states):
                states = None

    # Phase B: weight updates (only when RUN_CALIBRATION = True)
    c_pnls, c_sigs = [], []
    if RUN_CALIBRATION:
        c_prev_sig = None
        nan_count  = 0
        for i in range(WARMUP_BARS, c_end):
            x_b = torch.clamp(
                calib_X[i].unsqueeze(0).unsqueeze(0).to(DEVICE), -5.0, 5.0)
            r_b = calib_R[i].unsqueeze(0).unsqueeze(0).to(DEVICE)
            opt.zero_grad()
            sig, states = model(x_b, prev_states=states)
            states = [s.detach() for s in states]
            if torch.isnan(sig).any():
                nan_count += 1
                states = None
                continue
            loss = criterion(sig, r_b, prev_sig=c_prev_sig)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.10)
                opt.step()
            c_prev_sig = sig.squeeze(-1).detach()
            with torch.no_grad():
                c_pnls.append((sig.squeeze() * r_b.squeeze()).mean().item())
                c_sigs.append(sig.squeeze().cpu().numpy())

        if nan_count:
            print(f"  [warn] {nan_count} bars skipped due to NaN signal")

        if not c_sigs:
            print("  [warn] All calibration bars returned NaN -- feature mismatch?")
            c_sharpe = 0.0
        else:
            c_sig_arr = np.array(c_sigs)
            c_ret_arr = calib_R.numpy()[WARMUP_BARS:WARMUP_BARS + len(c_sigs)]
            c_sharpe  = sharpe(c_sig_arr.flatten(), c_ret_arr.flatten())
        print(f"  Done.  Avg PnL/bar: {np.mean(c_pnls)*100:.4f}%  |  Sharpe: {c_sharpe:.4f}")
    else:
        print("  Calibration skipped (RUN_CALIBRATION=False). Proceeding to backtest.")

    # Online-evolve backtest: weight update after every bar (true production mode)
    print(f"\n{'='*60}")
    print(f"ONLINE-EVOLVE BACKTEST  ({len(bt_X)} bars — weight update every bar)")
    print(f"{'='*60}\n")

    online_opt   = optim.AdamW(model.parameters(), lr=ONLINE_LR, weight_decay=1e-4)
    criterion_bt = RealPnLLoss()
    prev_sig_bt  = None
    b_sigs, b_rets = [], []
    nan_bt = 0

    for i in range(len(bt_X)):
        x_b = torch.clamp(
            bt_X[i].unsqueeze(0).unsqueeze(0).to(DEVICE), -5.0, 5.0)
        r_b = bt_R[i].unsqueeze(0).unsqueeze(0).to(DEVICE)

        online_opt.zero_grad()
        sig, states = model(x_b, prev_states=states)
        states = [s.detach() for s in states]

        if torch.isnan(sig).any():
            nan_bt += 1
            b_sigs.append(np.zeros(NUM_NODES, dtype=np.float32))
        else:
            loss = criterion_bt(sig, r_b, prev_sig=prev_sig_bt)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.10)
                online_opt.step()
            prev_sig_bt = sig.squeeze(-1).detach()
            b_sigs.append(sig.squeeze().detach().cpu().numpy())

        b_rets.append(bt_R[i].numpy())

    if nan_bt:
        print(f"  [warn] {nan_bt} bars had NaN signal")

    if not b_sigs:
        print("  No backtest bars (all data used in calibration)")
    else:
        bs = np.array(b_sigs); br = np.array(b_rets)
        pf_sharpe  = sharpe(bs.flatten(), br.flatten())
        gate_util  = (np.abs(bs) > 0.05).mean() * 100
        print(f"Portfolio Sharpe: {pf_sharpe:.4f}  |  Gate utilisation: {gate_util:.1f}%\n")
        print(f"  {'Pair':<8} {'Sharpe':>8} {'WinRate':>9} {'AvgSig':>8} {'CumPnL%':>10}")
        print(f"  {'-'*48}")
        for i, pair in enumerate(PAIRS):
            ps  = sharpe(bs[:, i], br[:, i])
            wr  = np.mean(np.sign(bs[:, i]) == np.sign(br[:, i])) * 100
            avg = np.abs(bs[:, i]).mean()
            pnl = (bs[:, i] * br[:, i]).sum() * 100
            gu  = (np.abs(bs[:, i]) > 0.05).mean() * 100
            print(f"  {pair:<8} {ps:>8.3f} {wr:>8.1f}% {avg:>8.4f} {pnl:>9.4f}%  gate={gu:.0f}%")

        # ── Plot Chart ────────────────────────────────────────────────────────
        try:
            import matplotlib.pyplot as plt
            plt.style.use('dark_background')
            plt.figure(figsize=(12, 6))
            
            # Use the datetime index for the backtest window
            bt_idx = common_idx[c_end:]
            for i, pair in enumerate(PAIRS):
                pair_pnl = np.cumsum(bs[:, i] * br[:, i]) * 100  # Cumulative %
                plt.plot(bt_idx, pair_pnl, label=f"{pair} (Sharpe {sharpe(bs[:, i], br[:, i]):.2f})")
                
            plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
            plt.title('TITAN-NL Walk-Forward Backtest PnL: 7-Day Simulation (15m bars)')
            plt.xlabel('Date / Time')
            plt.ylabel('Cumulative Return (%)')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            
            chart_path = os.path.join(_HERE, 'titan_backtest_chart.png')
            plt.savefig(chart_path, dpi=300)
            print(f"\n  Chart saved to: {chart_path}")
        except Exception as e:
            print(f"\n  [Warning] Could not plot chart: {e}")

    # ── Save calibrated state ─────────────────────────────────────────────
    torch.save(states, os.path.join(_HERE, 'titan_live_states_2026.pt'))
    torch.save(model.state_dict(), os.path.join(_HERE, 'titan_calibrated_2026.pth'))
    print(f"\nSaved: titan_calibrated_2026.pth  +  titan_live_states_2026.pt")
    print("TITAN-NL v5.0 Live 2026  COMPLETE.")
