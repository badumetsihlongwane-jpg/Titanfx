"""
TITAN-NL v6.1 — True Trading Policy Architecture
==================================================
New in v6.1:
  • Two-side return simulator (simulate_long_return & simulate_short_return)
  • Soft trade activation aligned with evaluation gating
  • PnL explicitly scaled to bps to battle regularization
  • Drastically reduced penalties, removed hold penalty
  • Strictly reset recurrent states to prevent overlapping chunk contamination
  • Metric evaluation strictly on executed trades (TP/SL/TO rates & MDD)
"""
import os, sys, io, json, pickle, math
from typing import Optional, Tuple, List, Dict
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass


# ── Architecture (inlined — fully self-contained, no titan.py needed) ─────────

class M3Optimizer(optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95, 0.999), eps=1e-8,
                 weight_decay=1e-2, ns_steps=5, slow_momentum_freq=10, alpha_slow=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        ns_steps=ns_steps, slow_momentum_freq=slow_momentum_freq, alpha_slow=alpha_slow)
        super().__init__(params, defaults)
        self.step_count = 0

    @staticmethod
    def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        if M.dim() != 2 or M.shape[0] > M.shape[1]: return M
        norm = M.norm()
        if norm < 1e-8: return M
        X = M / max(norm.item(), 1e-6)
        for _ in range(steps):
            A = X @ X.T
            if A.norm().item() > 1e4: return M
            X = 1.5 * X - 0.5 * A @ X
        return X * norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        self.step_count += 1
        for group in self.param_groups:
            beta1_fast, beta1_slow, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m1_fast'] = torch.zeros_like(p)
                    state['m1_slow'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['grad_running_avg'] = torch.zeros_like(p)
                    state['grad_count'] = 0
                state['step'] += 1
                m1_fast, m1_slow, v = state['m1_fast'], state['m1_slow'], state['v']
                m1_fast.mul_(beta1_fast).add_(grad, alpha=1 - beta1_fast)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                state['grad_count'] += 1
                cnt = state['grad_count']
                state['grad_running_avg'].mul_((cnt-1)/cnt).add_(grad, alpha=1./cnt)
                if self.step_count % group['slow_momentum_freq'] == 0:
                    m1_slow.mul_(beta1_slow).add_(state['grad_running_avg'], alpha=1-beta1_slow)
                    state['grad_running_avg'].zero_(); state['grad_count'] = 0
                bc1 = 1 - beta1_fast**state['step']; bc2 = 1 - beta2**state['step']
                m1c = m1_fast / bc1; vc = v / bc2
                m1o = self.newton_schulz(m1c, group['ns_steps']) if (m1c.dim()==2 and min(m1c.shape)>1) else m1c
                m1so= self.newton_schulz(m1_slow, group['ns_steps']) if (m1_slow.dim()==2 and min(m1_slow.shape)>1) else m1_slow
                combined = m1o + group['alpha_slow']*m1so
                p.data.addcdiv_(combined, vc.sqrt().add_(group['eps']), value=-group['lr'])
        return loss


class SelfModifyingDeltaMemory(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.proj_q  = nn.Linear(d_model, d_model, bias=False)
        self.proj_k  = nn.Linear(d_model, d_model, bias=False)
        self.proj_v  = nn.Linear(d_model, d_model, bias=False)
        self.value_generator = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.eta_proj  = nn.Sequential(nn.Linear(d_model, d_model//4), nn.SiLU(), nn.Linear(d_model//4, 1), nn.Sigmoid())
        self.alpha_proj= nn.Sequential(nn.Linear(d_model, d_model//4), nn.SiLU(), nn.Linear(d_model//4, 1), nn.Sigmoid())
        self.out_proj  = nn.Linear(d_model, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.register_buffer('init_memory', torch.zeros(d_model, d_model))

    def forward(self, x: torch.Tensor, prev_M=None):
        b, s, n, f = x.shape
        x_flat = x.view(b*n, s, f); residual = x_flat
        q = self.proj_q(x_flat); k = self.proj_k(x_flat); v = self.proj_v(x_flat)
        v_hat = self.value_generator(v)
        eta   = self.eta_proj(x_flat)  * 0.1 + 0.01
        alpha = self.alpha_proj(x_flat)* 0.5 + 0.5
        M = prev_M if prev_M is not None else self.init_memory.unsqueeze(0).expand(b*n,-1,-1).clone()
        outputs =[]
        for t in range(s):
            q_t = q[:,t,:]; k_t = F.normalize(k[:,t,:], dim=-1); v_t = v_hat[:,t,:]
            eta_t = eta[:,t,:].unsqueeze(-1); alpha_t = alpha[:,t,:].unsqueeze(-1)
            out_t = torch.bmm(M, q_t.unsqueeze(-1)).squeeze(-1)
            Mk = torch.bmm(M, k_t.unsqueeze(-1))
            M  = alpha_t*M - eta_t*torch.bmm(Mk, k_t.unsqueeze(-2)) + eta_t*torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(-2))
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
            nn.Sequential(nn.Linear(d_model, d_model*expansion), nn.SiLU(),
                          nn.Dropout(dropout), nn.Linear(d_model*expansion, d_model), nn.Dropout(dropout))
            for _ in range(self.num_levels)])
        self.level_weights = nn.Parameter(torch.ones(self.num_levels))
        self.level_norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_levels)])
        self.final_norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        b, s, n, f = x.shape
        x_flat = x.view(b*s*n, f)
        level_outputs =[]
        for li, (mlp, norm) in enumerate(zip(self.mlps, self.level_norms)):
            out = mlp(x_flat)
            if self.training:
                drop_p = 0.3*(1 - li/max(self.num_levels-1, 1))
                out = F.dropout(out, p=drop_p, training=True)
            level_outputs.append(norm(out + x_flat))
        weights = F.softmax(self.level_weights, dim=0)
        agg = sum(w*o for w, o in zip(weights, level_outputs))
        return self.final_norm(agg).view(b, s, n, f)


class MarketRegimeMemory(nn.Module):
    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes; self.d_model = d_model
        self.regime_detector = nn.Sequential(nn.Linear(d_model*2, d_model), nn.SiLU(), nn.Linear(d_model, 3), nn.Softmax(dim=-1))
        self.regime_eta   = nn.Parameter(torch.tensor([0.1, 0.05, 0.2]))
        self.regime_alpha = nn.Parameter(torch.tensor([0.8, 0.9, 0.6]))
        self.q_graph = nn.Linear(d_model, d_model)
        self.k_graph = nn.Linear(d_model, d_model)
        self.v_graph = nn.Linear(d_model, d_model)
        self.gate_net = nn.Sequential(nn.LayerNorm(d_model*3+3), nn.Linear(d_model*3+3, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate_net[-1].bias, 1.5)
        self.gate_act = nn.Sigmoid()
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        state = x[:, -3:, :, :].mean(dim=1)   # [B, N, D]
        b, n, d = state.shape; residual = state
        gmean = state.mean(dim=1, keepdim=True); gstd = state.std(dim=1, keepdim=True)
        regime_probs = self.regime_detector(torch.cat([state, gmean.expand(-1,n,-1)], dim=-1))
        gate_input   = torch.cat([state, gmean.expand(-1,n,-1), gstd.expand(-1,n,-1), regime_probs], dim=-1)
        alpha = self.gate_act(self.gate_net(gate_input))
        Q = self.q_graph(state); K = self.k_graph(state); V = self.v_graph(state)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d), dim=-1)
        I = torch.eye(n, device=x.device).unsqueeze(0).expand(b,-1,-1)
        mixed = alpha*I + (1-alpha)*attn
        out = self.norm(self.dropout(torch.matmul(mixed, V)) + residual)
        return out, alpha, attn, regime_probs


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PAIRS           =['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
NUM_NODES       = len(PAIRS)
CHUNK_LEN       = 16          # context window
MAX_HOLD_CAP    = 24          # max hold bars (24 × 30m = 12 h)
D_MODEL         = 128
NUM_LAYERS      = 3
CMS_CHUNK_SIZES = [16, 64, 256, 1024]
EPOCHS          = 75
PATIENCE        = 15
LR              = 1.5e-4
NOISE_STD       = 0.01
ONLINE_LR       = 5e-6
EXPLORATION_ONLY= True

# Date splits
TRAIN_START     = "2025-03-05"; TRAIN_END   = "2025-10-31"
VAL_START       = "2025-11-01"; VAL_END     = "2025-12-31"
CALIB_START     = "2026-01-01"; CALIB_END   = "2026-01-31"
BACKTEST_START  = "2026-02-01"; BACKTEST_END= "2026-03-04"

BARSPERYEAR_30M = 11088
BARSPERYEAR_15M = 22176
DATASET_INTERVAL= '30m'
BARSPERYEAR     = BARSPERYEAR_30M

# Loss weights (drastically reduced penalties)
SPREAD_BPS    = 1.0
LAMBDA_TURN   = 0.01
LAMBDA_CVAR   = 0.01
LAMBDA_GATE   = 2e-4
LAMBDA_SL     = 1e-4
LAMBDA_DIR    = 0.01
DIR_TARGET_SCALE = 600.0
OPPORTUNITY_BPS_FLOOR = 0.50
OPPORTUNITY_BPS_CAP = 8.0
LAMBDA_OPPORTUNITY = 0.002
CVAR_Q        = 0.10
GATE_THRESH   = 0.35
DIR_THRESH    = 0.03
SIZE_THRESH   = 0.02
TRADE_RATE_TARGET = 0.12
LAMBDA_TRADE_RATE = 0.02

# Dataset search
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()

DATASET_PATH = None
for _f in ('Titan30M_Dataset.csv', 'Titan15M_Dataset.csv', 'TitanForexDataset.csv'):
    for _d in[_here,
               '/kaggle/input/datasets/zackhlongwane/newawdat',
               '/kaggle/input/titanfx']:
        _p = os.path.join(_d, _f)
        if os.path.exists(_p):
            DATASET_PATH = _p
            break
    if DATASET_PATH:
        break
if not DATASET_PATH:
    DATASET_PATH = os.path.join(_here, 'Titan30M_Dataset.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"TITAN-NL v6.1: Policy Architecture | Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL — 6-HEAD POLICY
# ══════════════════════════════════════════════════════════════════════════════
class NestedGraphTitanV6(nn.Module):
    def __init__(self, num_nodes=NUM_NODES, feats_per_node=23,
                 d_model=D_MODEL, num_layers=NUM_LAYERS,
                 dropout=0.3, cms_chunk_sizes=CMS_CHUNK_SIZES):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model   = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(feats_per_node, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, CHUNK_LEN * 4, 1, d_model) * 0.02)

        self.temporal_layers = nn.ModuleList([
            SelfModifyingDeltaMemory(d_model, dropout) for _ in range(num_layers)
        ])
        self.cms = ContinuumMemoryMLP(d_model, cms_chunk_sizes, dropout=dropout)
        self.regime = MarketRegimeMemory(num_nodes, d_model, dropout=dropout)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        h = d_model // 2

        def _head(out=1, hidden=None):
            hid = hidden or h // 2
            return nn.Sequential(nn.Linear(h, hid), nn.GELU(), nn.Linear(hid, out))

        self.direction_head = _head()   # → tanh →[-1, 1]
        self.gate_head      = _head()   # → sigmoid → [0, 1]
        self.size_head      = _head()   # → sigmoid → [0, 1]
        self.tp_head        = _head()   # → 0.5 + 5.5·sigmoid → [0.5, 6.0] ATR
        self.sl_head        = _head()   # → 0.3 + 2.7·sigmoid →[0.3, 3.0] ATR
        self.hold_head      = _head()   # → 2 + 22·sigmoid → [2, 24] bars

        self._init_heads()

    def _init_heads(self):
        nn.init.zeros_(self.direction_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, 0.5)
        nn.init.zeros_(self.size_head[-1].bias)
        nn.init.constant_(self.tp_head[-1].bias, 0.0)
        nn.init.constant_(self.sl_head[-1].bias, -0.5)
        nn.init.constant_(self.hold_head[-1].bias, -1.0)

    def forward(self, x, prev_states=None, step=0):
        B, S, N, F_in = x.shape
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :S, :, :]

        if prev_states is None:
            prev_states = [None] * len(self.temporal_layers)
        new_states =[]
        for layer, p_M in zip(self.temporal_layers, prev_states):
            x, new_M = layer(x, p_M)
            new_states.append(new_M)

        x = self.cms(x, step)
        graph_out, alpha, attn_w, _ = self.regime(x)

        last_t   = x[:, -1, :, :]
        combined = last_t + graph_out
        trunk    = self.trunk(combined)

        direction = torch.tanh(self.direction_head(trunk)).squeeze(-1)
        gate      = torch.sigmoid(self.gate_head(trunk)).squeeze(-1)
        size      = torch.sigmoid(self.size_head(trunk)).squeeze(-1)
        tp_mult   = 0.5 + 5.5 * torch.sigmoid(self.tp_head(trunk)).squeeze(-1)
        sl_mult   = 0.3 + 2.7 * torch.sigmoid(self.sl_head(trunk)).squeeze(-1)
        hold_soft = 2.0 + 22.0 * torch.sigmoid(self.hold_head(trunk)).squeeze(-1)

        return {
            'direction': direction,
            'gate':      gate,
            'size':      size,
            'tp_mult':   tp_mult,
            'sl_mult':   sl_mult,
            'hold_bars': hold_soft,
            'states':    new_states,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRADE SIMULATORS (Two-Side Returns)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_long_return(
    tp_mult:      torch.Tensor,   # [B, N]
    sl_mult:      torch.Tensor,   # [B, N]
    hold_bars:    torch.Tensor,   # [B, N]
    entry_close:  torch.Tensor,   #[B, N]
    entry_atr:    torch.Tensor,   # [B, N]
    future_high:  torch.Tensor,   # [B, H, N]
    future_low:   torch.Tensor,   # [B, H, N]
    future_close: torch.Tensor,   # [B, H, N]
    spread_cost:  float = 0.0001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N = future_high.shape
    dev = entry_close.device

    entry = entry_close
    atr   = entry_atr.clamp(min=1e-8)

    tp_price = entry + tp_mult * atr
    sl_price = entry - sl_mult * atr

    hold_int = hold_bars.detach().round().long().clamp(1, H)

    tp_e = tp_price.unsqueeze(1).expand(B, H, N)
    sl_e = sl_price.unsqueeze(1).expand(B, H, N)
    fh, fl = future_high, future_low

    tp_hit_mask = fh >= tp_e
    sl_hit_mask = fl <= sl_e

    tp_only_mask = tp_hit_mask & ~sl_hit_mask

    bar_idx  = torch.arange(H, device=dev).view(1, H, 1).expand(B, H, N)
    hold_e   = hold_int.unsqueeze(1).expand(B, H, N)
    in_hold  = bar_idx < hold_e

    tp_valid = tp_only_mask & in_hold
    sl_valid = sl_hit_mask  & in_hold

    INF = H
    def _first(mask):
        has  = mask.any(dim=1)
        idx  = mask.float().argmax(dim=1)
        return torch.where(has, idx, torch.full_like(idx, INF))

    first_tp = _first(tp_valid)
    first_sl = _first(sl_valid)

    has_tp = (first_tp < INF)
    has_sl = (first_sl < INF)

    tp_wins  = has_tp & (~has_sl | (first_tp < first_sl))
    sl_wins  = has_sl & ~tp_wins
    timed_out= ~tp_wins & ~sl_wins

    ret_tp  = tp_mult * atr / (entry + 1e-8)
    ret_sl  = -sl_mult * atr / (entry + 1e-8)

    hold_last = (hold_int - 1).clamp(0, H - 1)
    fc_gather = future_close.gather(1, hold_last.unsqueeze(1).expand(B, 1, N)).squeeze(1)
    ret_timeout = (fc_gather - entry) / (entry + 1e-8)

    raw_return = (tp_wins.float()   * ret_tp
                + sl_wins.float()   * ret_sl
                + timed_out.float() * ret_timeout)

    raw_return = raw_return - spread_cost

    return raw_return, tp_wins.float(), sl_wins.float(), timed_out.float()


def simulate_short_return(
    tp_mult:      torch.Tensor,   # [B, N]
    sl_mult:      torch.Tensor,   #[B, N]
    hold_bars:    torch.Tensor,   # [B, N]
    entry_close:  torch.Tensor,   # [B, N]
    entry_atr:    torch.Tensor,   # [B, N]
    future_high:  torch.Tensor,   #[B, H, N]
    future_low:   torch.Tensor,   # [B, H, N]
    future_close: torch.Tensor,   # [B, H, N]
    spread_cost:  float = 0.0001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N = future_high.shape
    dev = entry_close.device

    entry = entry_close
    atr   = entry_atr.clamp(min=1e-8)

    tp_price = entry - tp_mult * atr
    sl_price = entry + sl_mult * atr

    hold_int = hold_bars.detach().round().long().clamp(1, H)

    tp_e = tp_price.unsqueeze(1).expand(B, H, N)
    sl_e = sl_price.unsqueeze(1).expand(B, H, N)
    fh, fl = future_high, future_low

    tp_hit_mask = fl <= tp_e
    sl_hit_mask = fh >= sl_e

    tp_only_mask = tp_hit_mask & ~sl_hit_mask

    bar_idx  = torch.arange(H, device=dev).view(1, H, 1).expand(B, H, N)
    hold_e   = hold_int.unsqueeze(1).expand(B, H, N)
    in_hold  = bar_idx < hold_e

    tp_valid = tp_only_mask & in_hold
    sl_valid = sl_hit_mask  & in_hold

    INF = H
    def _first(mask):
        has  = mask.any(dim=1)
        idx  = mask.float().argmax(dim=1)
        return torch.where(has, idx, torch.full_like(idx, INF))

    first_tp = _first(tp_valid)
    first_sl = _first(sl_valid)

    has_tp = (first_tp < INF)
    has_sl = (first_sl < INF)

    tp_wins  = has_tp & (~has_sl | (first_tp < first_sl))
    sl_wins  = has_sl & ~tp_wins
    timed_out= ~tp_wins & ~sl_wins

    ret_tp  = tp_mult * atr / (entry + 1e-8)
    ret_sl  = -sl_mult * atr / (entry + 1e-8)

    hold_last = (hold_int - 1).clamp(0, H - 1)
    fc_gather = future_close.gather(1, hold_last.unsqueeze(1).expand(B, 1, N)).squeeze(1)
    ret_timeout = -(fc_gather - entry) / (entry + 1e-8)

    raw_return = (tp_wins.float()   * ret_tp
                + sl_wins.float()   * ret_sl
                + timed_out.float() * ret_timeout)

    raw_return = raw_return - spread_cost

    return raw_return, tp_wins.float(), sl_wins.float(), timed_out.float()


# ══════════════════════════════════════════════════════════════════════════════
# 4. POLICY LOSS
# ══════════════════════════════════════════════════════════════════════════════
class TradingPolicyLoss(nn.Module):
    def __init__(self,
                 lambda_turn=LAMBDA_TURN, lambda_cvar=LAMBDA_CVAR,
                 lambda_gate=LAMBDA_GATE, lambda_sl=LAMBDA_SL,
                 lambda_dir=LAMBDA_DIR,
                 lambda_opp=LAMBDA_OPPORTUNITY,
                 dir_target_scale=DIR_TARGET_SCALE,
                 opportunity_bps_floor=OPPORTUNITY_BPS_FLOOR,
                 opportunity_bps_cap=OPPORTUNITY_BPS_CAP,
                 trade_rate_target=TRADE_RATE_TARGET,
                 lambda_trade_rate=LAMBDA_TRADE_RATE,
                 cvar_q=CVAR_Q, gate_thresh=GATE_THRESH,
                 dir_thresh=DIR_THRESH, size_thresh=SIZE_THRESH):
        super().__init__()
        self.lambda_turn = lambda_turn
        self.lambda_cvar = lambda_cvar
        self.lambda_gate = lambda_gate
        self.lambda_sl   = lambda_sl
        self.lambda_dir  = lambda_dir
        self.lambda_opp  = lambda_opp
        self.dir_target_scale = dir_target_scale
        self.opportunity_bps_floor = opportunity_bps_floor
        self.opportunity_bps_cap = opportunity_bps_cap
        self.trade_rate_target = trade_rate_target
        self.lambda_trade_rate = lambda_trade_rate
        self.cvar_q      = cvar_q
        self.gate_thresh = gate_thresh
        self.dir_thresh  = dir_thresh
        self.size_thresh = size_thresh

    def forward(self, action: Dict[str, torch.Tensor],
                ret_long: torch.Tensor,
                ret_short: torch.Tensor,
                prev_action: Optional[Dict] = None) -> torch.Tensor:

        direction = action['direction']
        gate      = action['gate']
        size      = action['size']
        sl_mult   = action['sl_mult']

        # Blend expected return
        p_long = 0.5 * (direction + 1.0)
        p_short = 1.0 - p_long
        expected_return = p_long * ret_long + p_short * ret_short
        edge = ret_long - ret_short

        # Soft trade activation aligned with eval
        gate_soft = torch.sigmoid(12.0 * (gate - self.gate_thresh))
        dir_soft  = torch.sigmoid(12.0 * (direction.abs() - self.dir_thresh))
        size_soft = torch.sigmoid(18.0 * (size - self.size_thresh))
        trade_soft = gate_soft * dir_soft * size_soft

        pos = trade_soft * size * direction.abs()

        # Core PnL scaled to bps
        pnl = pos * expected_return * 1e4
        loss_core = -pnl.mean()

        # CVaR tail penalty
        flat_pnl = pnl.view(-1)
        k = max(1, int(self.cvar_q * flat_pnl.numel()))
        worst = torch.topk(flat_pnl, k, largest=False).values
        cvar_pen = self.lambda_cvar * (-worst.mean())

        # Turnover penalty
        if prev_action is not None:
            turn_pen = self.lambda_turn * (
                (direction - prev_action['direction']).abs().mean() +
                (size      - prev_action['size']).abs().mean()
            )
        else:
            turn_pen = torch.tensor(0., device=direction.device)

        # Gate overtrading penalty
        gate_pen = self.lambda_gate * gate.mean()

        # SL sanity (no ultra-tight stops)
        sl_pen = self.lambda_sl * (1.0 / (sl_mult + 1e-6)).mean()

        # Directional alignment to market edge (prevents collapse to direction≈0)
        dir_target = torch.tanh(edge.detach() * self.dir_target_scale)
        dir_pen = self.lambda_dir * F.mse_loss(direction, dir_target)

        # Reward opening trades only where long/short edge is materially different
        opportunity_bps = edge.detach().abs() * 1e4
        opportunity = F.relu(opportunity_bps - self.opportunity_bps_floor)
        opportunity = torch.clamp(opportunity, max=self.opportunity_bps_cap)
        opp_bonus = self.lambda_opp * (pos * opportunity).mean()

        # Keep participation away from collapse (near 0%) and overtrading (near 100%)
        trade_rate = trade_soft.mean()
        trade_rate_pen = self.lambda_trade_rate * (trade_rate - self.trade_rate_target).pow(2)

        loss = loss_core + cvar_pen + turn_pen + gate_pen + sl_pen + dir_pen + trade_rate_pen - opp_bonus
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATASET
# ══════════════════════════════════════════════════════════════════════════════
class RollingWindowTradeDataset(Dataset):
    def __init__(self, features: np.ndarray,          
                 close_raw: np.ndarray,               
                 atr_raw:   np.ndarray,               
                 high_raw:  np.ndarray,               
                 low_raw:   np.ndarray,               
                 chunk_len: int = CHUNK_LEN,
                 max_horizon: int = MAX_HOLD_CAP):
        self.X  = torch.FloatTensor(features)
        self.C  = torch.FloatTensor(close_raw)
        self.A  = torch.FloatTensor(atr_raw)
        self.H  = torch.FloatTensor(high_raw)
        self.L  = torch.FloatTensor(low_raw)
        self.chunk_len   = chunk_len
        self.max_horizon = max_horizon
        T = features.shape[0]
        self.valid_t = list(range(chunk_len - 1, T - max_horizon))

    def __len__(self):
        return len(self.valid_t)

    def __getitem__(self, idx):
        t = self.valid_t[idx]
        s = t - self.chunk_len + 1

        x_window    = self.X[s : t+1]              
        entry_close = self.C[t]                     
        entry_atr   = self.A[t]                     
        future_high = self.H[t+1 : t+1+self.max_horizon]  
        future_low  = self.L[t+1 : t+1+self.max_horizon]  
        future_close= self.C[t+1 : t+1+self.max_horizon]  

        return x_window, entry_close, entry_atr, future_high, future_low, future_close


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_titan_dataset_v6(path: str):
    print(f"\n>>> Loading {os.path.basename(path)} ...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"    Shape: {df.shape} | {df.index[0]} → {df.index[-1]}")

    pairs = []
    for col in df.columns:
        for p in['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']:
            if col.startswith(p + '_') and p not in pairs:
                pairs.append(p)
    if not pairs:
        pairs =['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    print(f"    Pairs: {pairs}")

    macro_cols =[c for c in df.columns
                  if not any(c.startswith(p+'_') for p in pairs)]

    node_arrays, close_arr_list, atr_arr_list = [], [], []
    high_arr_list,  low_arr_list = [],[]
    node_col_map = {}

    for pair in pairs:
        pair_cols =[c for c in df.columns if c.startswith(pair + '_')]
        feat_cols = pair_cols + macro_cols
        node_col_map[pair] = feat_cols

        mat = df[feat_cols].values.astype(np.float32)
        node_arrays.append(mat)

        def _col(suffix):
            c = f'{pair}_{suffix}'
            return df[c].values.astype(np.float32) if c in df.columns else np.ones(len(df), dtype=np.float32)

        close_arr_list.append(_col('Close'))
        high_arr_list.append(_col('High'))
        low_arr_list.append(_col('Low'))

        if f'{pair}_atr14raw' in df.columns:
            atr_arr_list.append(df[f'{pair}_atr14raw'].values.astype(np.float32))
        elif f'{pair}_atr14n' in df.columns:
            atr_arr_list.append((df[f'{pair}_atr14n'] * df[f'{pair}_Close']).values.astype(np.float32))
        else:
            lr = np.log(df[f'{pair}_Close'] / df[f'{pair}_Close'].shift(1)).fillna(0)
            atr_arr_list.append((lr.rolling(14).std().fillna(method='bfill') *
                                 df[f'{pair}_Close']).values.astype(np.float32))

    T = len(df)
    N = len(pairs)
    feats_per_node = node_arrays[0].shape[1]
    master = np.stack(node_arrays, axis=1)

    close_raw = np.stack(close_arr_list, axis=1)
    high_raw  = np.stack(high_arr_list,  axis=1)
    low_raw   = np.stack(low_arr_list,   axis=1)
    atr_raw   = np.stack(atr_arr_list,   axis=1)

    dates = df.index
    schema = {
        'feats_per_node': feats_per_node,
        'pairs': pairs,
        'node_cols': node_col_map,
        'macro_cols': macro_cols,
    }

    print(f"    Master: {master.shape} | feats/node: {feats_per_node}")
    return master, close_raw, atr_raw, high_raw, low_raw, feats_per_node, dates, schema


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch_v6(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    total_loss = 0.0
    prev_action = None
    step = 0

    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    for x_win, e_close, e_atr, f_high, f_low, f_close in loader:
        # Strictly reset recurrent state across batched overlapping windows
        prev_states = None

        x_win   = x_win.to(device)
        e_close = e_close.to(device)
        e_atr   = e_atr.to(device)
        f_high  = f_high.to(device)
        f_low   = f_low.to(device)
        f_close = f_close.to(device)

        x_win = torch.clamp(x_win + torch.randn_like(x_win) * NOISE_STD, -10, 10)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                act = model(x_win, prev_states, step)
                ret_long, _, _, _ = simulate_long_return(
                    act['tp_mult'], act['sl_mult'], act['hold_bars'],
                    e_close, e_atr, f_high, f_low, f_close)
                ret_short, _, _, _ = simulate_short_return(
                    act['tp_mult'], act['sl_mult'], act['hold_bars'],
                    e_close, e_atr, f_high, f_low, f_close)
                
                loss = criterion(act, ret_long, ret_short, prev_action)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            act = model(x_win, prev_states, step)
            ret_long, _, _, _ = simulate_long_return(
                act['tp_mult'], act['sl_mult'], act['hold_bars'],
                e_close, e_atr, f_high, f_low, f_close)
            ret_short, _, _, _ = simulate_short_return(
                act['tp_mult'], act['sl_mult'], act['hold_bars'],
                e_close, e_atr, f_high, f_low, f_close)
            
            loss = criterion(act, ret_long, ret_short, prev_action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        prev_action  = {k: v.detach() for k, v in act.items() if k != 'states'}
        total_loss  += loss.item()
        step        += 1

    return total_loss / max(step, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION (trade-based metrics)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_v6(model, loader, criterion, device, periods_per_year=BARSPERYEAR_30M):
    model.eval()
    total_loss = 0.0
    step = 0

    all_realized, all_mask     = [],[]
    all_tp, all_sl, all_to     = [], [],[]
    all_gate, all_size         = [],[]
    all_tp_m, all_sl_m, all_h  = [], [],[]

    with torch.no_grad():
        for x_win, e_close, e_atr, f_high, f_low, f_close in loader:
            x_win   = x_win.to(device)
            e_close = e_close.to(device)
            e_atr   = e_atr.to(device)
            f_high  = f_high.to(device)
            f_low   = f_low.to(device)
            f_close = f_close.to(device)
            x_win   = torch.clamp(x_win, -10, 10)

            act = model(x_win, prev_states=None)
            ret_long, tp_l, sl_l, to_l = simulate_long_return(
                act['tp_mult'], act['sl_mult'], act['hold_bars'],
                e_close, e_atr, f_high, f_low, f_close)
            ret_short, tp_s, sl_s, to_s = simulate_short_return(
                act['tp_mult'], act['sl_mult'], act['hold_bars'],
                e_close, e_atr, f_high, f_low, f_close)

            loss = criterion(act, ret_long, ret_short)

            # Route metrics based on chosen direction
            is_long = (act['direction'] >= 0).float()
            is_short = 1.0 - is_long

            realized = is_long * ret_long + is_short * ret_short
            tp_h = is_long * tp_l + is_short * tp_s
            sl_h = is_long * sl_l + is_short * sl_s
            to_h = is_long * to_l + is_short * to_s

            trade_mask = ((act['gate'] > GATE_THRESH) &
                          (act['direction'].abs() > DIR_THRESH) &
                          (act['size'] > SIZE_THRESH)).float()
            exec_weight = trade_mask * act['size'] * act['direction'].abs()
            realized_exec = realized * exec_weight

            all_realized.append(realized_exec.cpu().numpy())
            all_mask.append(trade_mask.cpu().numpy())
            all_tp.append(tp_h.cpu().numpy())
            all_sl.append(sl_h.cpu().numpy())
            all_to.append(to_h.cpu().numpy())
            all_gate.append(act['gate'].cpu().numpy())
            all_size.append(act['size'].cpu().numpy())
            all_tp_m.append(act['tp_mult'].cpu().numpy())
            all_sl_m.append(act['sl_mult'].cpu().numpy())
            all_h.append(act['hold_bars'].cpu().numpy())

            total_loss += loss.item()
            step += 1

    def cat(lst): return np.concatenate([a.reshape(-1) for a in lst])

    realized_f = cat(all_realized)
    mask_f     = cat(all_mask)
    trades_idx = mask_f > 0.5
    trades     = realized_f[trades_idx]

    if len(trades) == 0:
        sharpe = 0.0
    else:
        mu  = trades.mean()
        std = trades.std() + 1e-9
        sharpe = mu / std * math.sqrt(periods_per_year)

    win_rate   = (trades > 0).mean() * 100 if len(trades) else 0
    wins       = trades[trades > 0]
    losses     = trades[trades < 0]
    pf         = wins.sum() / (-losses.sum() + 1e-9) if len(losses) else float('inf')
    total_ret  = trades.sum() * 100   

    # Max drawdown correctly computed on actual executed PnL path
    executed_pnl_path = realized_f * mask_f
    cum = np.cumsum(executed_pnl_path)
    peak = np.maximum.accumulate(cum)
    mdd  = ((peak - cum) / (np.abs(peak) + 1e-9)).max() * 100

    tp_f = cat(all_tp)
    sl_f = cat(all_sl)
    to_f = cat(all_to)

    # Base rates strictly on actual traded subsets
    if len(trades) > 0:
        tp_rate = tp_f[trades_idx].mean() * 100
        sl_rate = sl_f[trades_idx].mean() * 100
        to_rate = to_f[trades_idx].mean() * 100
    else:
        tp_rate = sl_rate = to_rate = 0.0

    gate_util= mask_f.mean() * 100
    avg_size = cat(all_size).mean()
    avg_tp_m = cat(all_tp_m).mean()
    avg_sl_m = cat(all_sl_m).mean()
    avg_hold = cat(all_h).mean()

    metrics = dict(
        loss=total_loss / max(step, 1),
        sharpe=sharpe, total_ret=total_ret, mdd=mdd,
        win_rate=win_rate, pf=pf,
        tp_rate=tp_rate, sl_rate=sl_rate, to_rate=to_rate,
        gate_util=gate_util, avg_size=avg_size,
        avg_tp_m=avg_tp_m, avg_sl_m=avg_sl_m, avg_hold=avg_hold,
        n_trades=len(trades),
    )
    return metrics


def print_metrics(tag, m):
    print(f"\n  {tag}")
    print(f"    Loss={m['loss']:.5f}  Sharpe={m['sharpe']:.2f}  "
          f"TotalRet={m['total_ret']:.3f}%  MDD={m['mdd']:.2f}%")
    print(f"    WinRate={m['win_rate']:.1f}%  PF={m['pf']:.2f}  "
          f"Trades={m['n_trades']}")
    print(f"    TP={m['tp_rate']:.1f}%  SL={m['sl_rate']:.1f}%  "
          f"TO={m['to_rate']:.1f}%  GateUtil={m['gate_util']:.1f}%")
    print(f"    AvgSize={m['avg_size']:.3f}  "
          f"AvgTP={m['avg_tp_m']:.2f}x  AvgSL={m['avg_sl_m']:.2f}x  "
          f"AvgHold={m['avg_hold']:.1f}bars")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SHARPE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def sharpe_from_trades(trade_returns, bpy=BARSPERYEAR_30M):
    if len(trade_returns) < 2:
        return 0.0
    mu  = np.mean(trade_returns)
    std = np.std(trade_returns) + 1e-9
    return float(mu / std * math.sqrt(bpy))


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN — TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    (master, close_raw, atr_raw, high_raw, low_raw,
     feats_per_node, dates, schema) = load_titan_dataset_v6(DATASET_PATH)

    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_mask   = (dates >= VAL_START)   & (dates <= VAL_END)
    calib_mask = (dates >= CALIB_START) & (dates <= CALIB_END)
    back_mask  = (dates >= BACKTEST_START) & (dates <= BACKTEST_END)

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    calib_idx = np.where(calib_mask)[0]
    back_idx  = np.where(back_mask)[0]

    print(f"\n  Splits — Train:{len(train_idx):,}  Val:{len(val_idx):,}  "
          f"Calib:{len(calib_idx):,}  Backtest:{len(back_idx):,}")

    N, Nodes, Feats = master.shape
    scaler = RobustScaler().fit(master[train_idx].reshape(-1, Feats))
    scaled = np.nan_to_num(
        scaler.transform(master.reshape(-1, Feats)).reshape(N, Nodes, Feats),
        nan=0., posinf=5., neginf=-5.)

    with open('titan_v6_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('titan_v6_schema.json', 'w') as f:
        json.dump({**schema, 'chunk_len': CHUNK_LEN,
                   'max_hold_cap': MAX_HOLD_CAP, 'd_model': D_MODEL,
                   'num_layers': NUM_LAYERS, 'cms': CMS_CHUNK_SIZES}, f, indent=2)
    print("  titan_v6_scaler.pkl + titan_v6_schema.json saved")

    def _mk(idx):
        ds = RollingWindowTradeDataset(
            scaled[idx], close_raw[idx], atr_raw[idx],
            high_raw[idx], low_raw[idx])
        return DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    def _mk1(idx):   
        ds = RollingWindowTradeDataset(
            scaled[idx], close_raw[idx], atr_raw[idx],
            high_raw[idx], low_raw[idx])
        return DataLoader(ds, batch_size=1, shuffle=False)

    train_loader = _mk(train_idx)
    val_loader   = _mk1(val_idx)
    calib_loader = _mk1(calib_idx)
    back_loader  = _mk1(back_idx)

    print(f"  Train batches: {len(train_loader)}  "
          f"Val samples: {len(val_loader.dataset)}")

    model = NestedGraphTitanV6(
        num_nodes=NUM_NODES, feats_per_node=feats_per_node,
        d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=0.3,
        cms_chunk_sizes=CMS_CHUNK_SIZES).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_p:,}")

    criterion = TradingPolicyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

    use_amp = DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()

    print(f"\n{'='*60}")
    print(f"PHASE 1: Training  ({EPOCHS} epochs, patience={PATIENCE})")
    print(f"{'='*60}")

    best_sharpe  = -1e9
    no_improve   = 0
    save_name    = 'Best_TITANv6_EXPLORATION.pth' if EXPLORATION_ONLY else 'Best_TITANv6_EVOLVING.pth'

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch_v6(
            model, train_loader, criterion, optimizer, DEVICE, use_amp=use_amp)
        scheduler.step()

        val_m = evaluate_v6(model, val_loader, criterion, DEVICE,
                            periods_per_year=BARSPERYEAR)
        sharpe = val_m['sharpe']

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"TrLoss={tr_loss:.5f}  ValLoss={val_m['loss']:.5f} | "
              f"ValSharpe={sharpe:.4f}  "
              f"WinRate={val_m['win_rate']:.1f}%  "
              f"Trades={val_m['n_trades']}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            no_improve  = 0
            torch.save(model.state_dict(), save_name)
            print(f"  >> Best model saved → {save_name}  (Sharpe={sharpe:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"PHASE 2: Walk-Forward Calibration + Backtest")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(save_name, map_location=DEVICE, weights_only=True))

    print(f"  [1/2] Calibration fine-tune ({CALIB_START} – {CALIB_END})...")
    calib_opt = optim.AdamW(model.parameters(), lr=ONLINE_LR * 10, weight_decay=1e-4)
    prev_act  = None
    for x_win, e_close, e_atr, f_high, f_low, f_close in calib_loader:
        x_win, e_close, e_atr = x_win.to(DEVICE), e_close.to(DEVICE), e_atr.to(DEVICE)
        f_high, f_low, f_close = f_high.to(DEVICE), f_low.to(DEVICE), f_close.to(DEVICE)
        model.train(); calib_opt.zero_grad()

        act = model(x_win, prev_states=None)
        ret_long, _, _, _ = simulate_long_return(
            act['tp_mult'], act['sl_mult'], act['hold_bars'],
            e_close, e_atr, f_high, f_low, f_close)
        ret_short, _, _, _ = simulate_short_return(
            act['tp_mult'], act['sl_mult'], act['hold_bars'],
            e_close, e_atr, f_high, f_low, f_close)

        loss = criterion(act, ret_long, ret_short, prev_act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        calib_opt.step()
        prev_act= {k: v.detach() for k, v in act.items() if k != 'states'}

    model.eval()
    print(f"\n  [2/2] Backtest ({BACKTEST_START} – {BACKTEST_END})...")
    bt_m = evaluate_v6(model, back_loader, criterion, DEVICE, periods_per_year=BARSPERYEAR)
    print_metrics("BACKTEST", bt_m)

    final_name = 'Best_TITANv6_EVOLVING.pth'
    torch.save(model.state_dict(), final_name)
    print(f"\n  {final_name}  saved")
    print(f"\n{'='*60}")
    print("TITAN-NL v6.1 TRAINING COMPLETE")
    print("Download: Best_TITANv6_EVOLVING.pth  titan_v6_scaler.pkl  titan_v6_schema.json")
    print(f"{'='*60}")
