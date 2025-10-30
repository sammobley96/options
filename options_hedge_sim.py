import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from datetime import datetime, timedelta

st.set_page_config(page_title="Options Hedge Optimizer — Surface v10", layout="wide")
plt.rcParams["axes.unicode_minus"] = False

# ---------- Black-Scholes ----------
def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "C" else max(0.0, K - S)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "C":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, sigma, option_type):
    if T <= 0 or sigma <= 0:
        if option_type == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (log(S / K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1.0

# ---------- Helpers ----------
def pick_exp(exp_list, target_days, today):
    parsed = []
    for d in exp_list:
        dt = datetime.strptime(d, "%Y-%m-%d")
        days = (dt - today).days
        if days > 0:
            parsed.append((d, days))
    if not parsed:
        return None
    return min(parsed, key=lambda x: abs(x[1] - target_days))[0]

def nearest_strike(strikes_series, target):
    idx = (strikes_series - target).abs().argsort().iloc[0]
    return float(strikes_series.iloc[idx])

def atm_iv_for_exp(ticker, exp, spot):
    try:
        ch = ticker.option_chain(exp)
        calls = ch.calls.dropna(subset=["impliedVolatility", "strike"])
        if calls.empty:
            return None
        calls["dist"] = (calls["strike"] - spot).abs()
        ivs = calls.nsmallest(5, "dist")["impliedVolatility"].values
        return float(np.median(ivs) * 100) if ivs.size > 0 else None
    except Exception:
        return None

def auto_iv_crush_estimate(ticker, spot, days_short, today):
    """Estimate IV crush from term structure (ATM IV short vs ~4x days)."""
    try:
        exps = ticker.options
        if not exps:
            return None, None, None
        short_exp = pick_exp(exps, days_short, today)
        long_exp  = pick_exp(exps, max(int(days_short * 4), days_short + 21), today)
        iv_short = atm_iv_for_exp(ticker, short_exp, spot) if short_exp else None
        iv_long  = atm_iv_for_exp(ticker, long_exp,  spot) if long_exp  else None
        if iv_short and iv_long and iv_short > 0:
            crush = max(0.0, min(((iv_short - iv_long) / iv_short) * 100.0, 60.0))
            return crush, short_exp, long_exp
        return None, short_exp, long_exp
    except Exception:
        return None, None, None

def scaled_call_iv_after_crush(call_iv_now_pct, iv_crush_short_pct,
                               call_exp_str, risk_window_days, today, ticker, spot):
    """
    Scale short-term 'event' IV crush for a longer-dated call.
    - Time scaling via sqrt(risk_window_days / days_to_call_expiry), clipped [0,1]
    - Floor at a longer-term ATM IV anchor (≈ 4× risk window)
    Returns: new_call_iv_pct, details
    """
    d_call = max(1, (datetime.strptime(call_exp_str, "%Y-%m-%d") - today).days)
    frac_time = min(1.0, max(0.0, (risk_window_days / d_call) ** 0.5))

    long_anchor_days = max(risk_window_days * 4, risk_window_days + 21)
    try:
        exps = ticker.options
        anchor_exp = pick_exp(exps, int(long_anchor_days), today)
        iv_floor = atm_iv_for_exp(ticker, anchor_exp, spot) if anchor_exp else None
    except Exception:
        anchor_exp, iv_floor = None, None

    crush_pts_short = (call_iv_now_pct * (iv_crush_short_pct / 100.0))
    crush_pts_call = crush_pts_short * frac_time
    iv_new = max(1e-6, call_iv_now_pct - crush_pts_call)
    if iv_floor is not None:
        iv_new = max(iv_new, iv_floor)

    info = {
        "days_to_call_exp": d_call,
        "frac_time": frac_time,
        "crush_pts_short": crush_pts_short,
        "crush_pts_call": crush_pts_call,
        "iv_floor": iv_floor,
        "iv_new": iv_new,
    }
    return iv_new, info

# ---------- Surface builders ----------
def build_iv_curve_for_exp(ticker, exp, spot, kind="calls_and_puts"):
    """
    Build a smooth IV curve vs log-moneyness (ln(K/spot)) for a given expiration.
    Uses PCHIP (shape-preserving) to avoid oscillations.
    kind: "puts", "calls", or "calls_and_puts"
    Returns: (interp_fn, m_grid, iv_grid_pct)
    """
    ch = ticker.option_chain(exp)
    frames = []
    if kind in ("calls", "calls_and_puts"):
        c = ch.calls.dropna(subset=["strike", "impliedVolatility"])
        if not c.empty:
            frames.append(c[["strike", "impliedVolatility"]].assign(side="C"))
    if kind in ("puts", "calls_and_puts"):
        p = ch.puts.dropna(subset=["strike", "impliedVolatility"])
        if not p.empty:
            frames.append(p[["strike", "impliedVolatility"]].assign(side="P"))

    if not frames:
        return None, None, None

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna()
    if df.empty:
        return None, None, None

    df["mny"] = np.log(df["strike"].astype(float) / float(spot))
    df = df.sort_values("mny")
    df = df.groupby("mny", as_index=False)["impliedVolatility"].median()  # de-dup

    x = df["mny"].values
    y = (df["impliedVolatility"].values * 100.0)  # to %
    if len(x) < 3:
        return None, x, y

    interp = PchipInterpolator(x, y, extrapolate=True)
    return interp, x, y

def iv_from_surface(interp, S_new, K):
    """Return IV% from the curve at the log-moneyness of K vs S_new."""
    if interp is None:
        return None
    mny = np.log(float(K) / float(S_new))
    return float(np.maximum(1e-6, interp(mny)))

# ---------- UI ----------
st.title("🎯 Options Hedge Optimizer — Surface-based (v10)")

left, right = st.columns(2)
with left:
    symbol = st.text_input("Underlying", "MSFT").upper()
    call_strike = st.number_input("Your Call Strike", value=565.0)
    call_exp = st.text_input("Your Call Expiration (YYYY-MM-DD)", "2025-12-19")
    num_calls = st.number_input("Number of Call Contracts", value=3, step=1)
with right:
    call_cost_basis = st.number_input("Avg Cost per Call ($)", value=15.02)
    days_until_exit = st.slider("Days Until Exit (P/L snapshot horizon)", 1, 30, 1)
    custom_hedge_ratio = st.slider("Custom Hedge %", 0, 100, 98)

st.markdown("#### Scenario — what are you hedging against?")
sc1, sc2, sc3 = st.columns([1,1,1])
with sc1:
    drop_pct = st.slider("💥 % Drop to Hedge", 1, 20, 5)
with sc2:
    risk_window_days = st.slider("⏳ Risk Window (days)", 1, 30, 2)
with sc3:
    expected_upside_pct = st.slider("🚀 Expected Upside (%)", 0, 20, 5)

st.markdown("#### Volatility & Surface Dynamics")
v1, v2, v3 = st.columns([1,1,1])
with v1:
    use_auto_iv = st.checkbox("Use automatic IV calibration (term crush)", value=True)
with v2:
    manual_iv_crush = st.slider("Manual IV Crush (%) (if auto unavailable)", 0, 60, 15)
with v3:
    dyn_mode = st.selectbox("IV surface dynamics", ["Sticky-Delta (preserve moneyness)", "Sticky-Strike (preserve strike)"])

r = 0.05
today = datetime.today()

# ---------- Spot ----------
ticker = yf.Ticker(symbol)
hist = ticker.history(period="1d")
if hist.empty:
    st.error("Could not fetch underlying price.")
    st.stop()
S0 = float(hist["Close"].iloc[-1])
st.caption(f"Spot S0 = ${S0:.2f}")

# ---------- Your call ----------
try:
    chain_call = ticker.option_chain(call_exp)
except Exception as e:
    st.error(f"Could not fetch your call chain for {call_exp}: {e}")
    st.stop()

calls_df = chain_call.calls
call_row = calls_df[calls_df["strike"] == call_strike]
if call_row.empty:
    st.error(f"No {symbol} {call_strike} C for {call_exp}.")
    st.stop()

call_mid = float((call_row["bid"].iloc[0] + call_row["ask"].iloc[0]) / 2)
call_iv_now = float(call_row["impliedVolatility"].iloc[0] * 100)
st.caption(f"Your call mid ≈ ${call_mid:.2f}, IV ≈ {call_iv_now:.1f}%")

# ---------- Scenario-driven hedge (expiry & strikes) ----------
target_exp_days = int(risk_window_days + 2)  # small buffer
exps = ticker.options
if not exps:
    st.error("No option expirations available.")
    st.stop()

hedge_exp = pick_exp(exps, target_exp_days, today)
if hedge_exp is None:
    st.error("Could not find a suitable hedge expiration.")
    st.stop()

try:
    chain_hedge = ticker.option_chain(hedge_exp)
except Exception as e:
    st.error(f"Could not fetch hedge chain for {hedge_exp}: {e}")
    st.stop()

puts_df = chain_hedge.puts.dropna(subset=["strike", "bid", "ask", "impliedVolatility"])
if puts_df.empty:
    st.error(f"No puts available for {hedge_exp}.")
    st.stop()

long_target = S0 * (1.0 - drop_pct / 100.0)
width_dollars = S0 * (drop_pct / 2.0) / 100.0
short_target = long_target - width_dollars

long_strike = nearest_strike(puts_df["strike"], long_target)
short_strike = nearest_strike(puts_df["strike"], short_target)

row_long = puts_df[puts_df["strike"] == long_strike].iloc[0]
row_short = puts_df[puts_df["strike"] == short_strike].iloc[0]

long_mid = float((row_long["bid"] + row_long["ask"]) / 2.0)
short_mid = float((row_short["bid"] + row_short["ask"]) / 2.0)
spread_cost_live = long_mid - short_mid
put_iv_now = float(row_long["impliedVolatility"] * 100)

st.markdown(
    f"**Scenario hedge:** target exp ~ {target_exp_days}d → **{hedge_exp}** | "
    f"strikes **{long_strike:.0f}/{short_strike:.0f}** | live spread ≈ **${spread_cost_live:.2f}** | "
    f"put IV ≈ **{put_iv_now:.1f}%**"
)

# ---------- Auto IV crush ----------
if use_auto_iv:
    auto_crush, short_exp_used, long_exp_used = auto_iv_crush_estimate(
        ticker, S0, days_short=target_exp_days, today=today
    )
    if auto_crush is not None:
        iv_crush = auto_crush
        st.success(
            f"📉 Auto IV Crush ≈ {iv_crush:.1f}% "
            + (f"(term: {short_exp_used} → {long_exp_used})" if short_exp_used and long_exp_used else "")
        )
    else:
        iv_crush = float(manual_iv_crush)
        st.warning("Auto IV crush unavailable — using manual setting.")
else:
    iv_crush = float(manual_iv_crush)
    st.info(f"Manual IV Crush: {iv_crush:.1f}%")

# ---------- Build IV surfaces for call & hedge expiries ----------
call_curve, _, _  = build_iv_curve_for_exp(ticker, call_exp,  S0, kind="calls_and_puts")
hedge_curve, _, _ = build_iv_curve_for_exp(ticker, hedge_exp, S0, kind="calls_and_puts")

def scenario_iv_from_curve(curve, S_ref, S_scn, K, mode):
    if curve is None:
        return None
    if mode.startswith("Sticky-Delta"):  # preserve moneyness vs new spot
        return iv_from_surface(curve, S_scn, K)
    else:  # Sticky-Strike: preserve strike moneyness vs original spot
        return iv_from_surface(curve, S_ref, K)

def atm_iv_from_curve(curve, S_ref):
    if curve is None:
        return None
    return iv_from_surface(curve, S_ref, S_ref)

# ATM levels now (for crush shifting)
call_atm_now  = atm_iv_from_curve(call_curve,  S0) if call_curve else call_iv_now
hedge_atm_now = atm_iv_from_curve(hedge_curve, S0) if hedge_curve else put_iv_now

# Scaled crush for CALL ATM; full crush for hedge ATM
scaled_call_atm_after, _ivinfo = scaled_call_iv_after_crush(
    call_iv_now_pct = call_atm_now if call_atm_now else call_iv_now,
    iv_crush_short_pct = iv_crush,
    call_exp_str = call_exp,
    risk_window_days = risk_window_days,
    today = today, ticker = ticker, spot = S0
)
hedge_atm_after = max(1e-6, (hedge_atm_now if hedge_atm_now else put_iv_now) * (1.0 - iv_crush/100.0))

def crush_shifted_iv(base_iv, atm_now, atm_after):
    if base_iv is None:
        return None
    if (atm_now is None) or (atm_after is None):
        return max(1e-6, base_iv)
    return max(1e-6, base_iv + (atm_after - atm_now))

# ---------- Simulation engine (surface-based IV) ----------
def simulate_ratio_surface(hedge_ratio):
    # Size hedge contracts by delta coverage (current greeks)
    T_call = max(0.0, (datetime.strptime(call_exp, "%Y-%m-%d") - today).days / 365.0)
    call_delta_now = bs_delta(S0, call_strike, T_call, call_iv_now / 100.0, "C")
    total_call_delta = call_delta_now * num_calls * 100.0

    T_put = max(0.0, (datetime.strptime(hedge_exp, "%Y-%m-%d") - today).days / 365.0)
    put_delta_now = abs(bs_delta(S0, long_strike, T_put, put_iv_now / 100.0, "P")) * 100.0

    hedge_contracts = max(1, int(round((total_call_delta * (hedge_ratio / 100.0)) / max(put_delta_now, 1e-9))))

    # Price grid near spot
    prices = np.arange(S0 * 0.9, S0 * 1.1 + 1e-9, 2.0)

    # Exit-time T (after your snapshot horizon)
    T_exit_call = max(0.0, (datetime.strptime(call_exp, "%Y-%m-%d") - today - timedelta(days=days_until_exit)).days / 365.0)
    T_exit_put  = max(0.0, (datetime.strptime(hedge_exp, "%Y-%m-%d") - today - timedelta(days=days_until_exit)).days / 365.0)

    call_vals, put_up_vals, put_lo_vals = [], [], []

    for S_scn in prices:
        # Pull base IVs from surfaces per scenario & dynamics
        base_iv_call = scenario_iv_from_curve(call_curve, S0, S_scn, call_strike, dyn_mode)
        base_iv_up   = scenario_iv_from_curve(hedge_curve, S0, S_scn, long_strike, dyn_mode)
        base_iv_lo   = scenario_iv_from_curve(hedge_curve, S0, S_scn, short_strike, dyn_mode)

        # Apply ATM-level crush shift (shape preserved)
        iv_call_scn = crush_shifted_iv(base_iv_call if base_iv_call else call_iv_now, call_atm_now, scaled_call_atm_after)
        iv_up_scn   = crush_shifted_iv(base_iv_up   if base_iv_up   else put_iv_now,  hedge_atm_now, hedge_atm_after)
        iv_lo_scn   = crush_shifted_iv(base_iv_lo   if base_iv_lo   else put_iv_now,  hedge_atm_now, hedge_atm_after)

        call_vals.append(bs_price(S_scn, call_strike, T_exit_call, r, iv_call_scn/100.0, "C"))
        put_up_vals.append(bs_price(S_scn, long_strike,  T_exit_put, r, iv_up_scn/100.0,   "P"))
        put_lo_vals.append(bs_price(S_scn, short_strike, T_exit_put, r, iv_lo_scn/100.0,   "P"))

    call_val = np.array(call_vals)
    put_up   = np.array(put_up_vals)
    put_lo   = np.array(put_lo_vals)

    call_ret  = (call_val - call_cost_basis) * num_calls * 100.0
    hedge_ret = hedge_contracts * ((put_up - put_lo) - spread_cost_live) * 100.0
    total_pl  = call_ret + hedge_ret

    # Sanity snapshot at +0.8%
    target_move_pct = 0.8
    S_target = S0 * (1 + target_move_pct / 100.0)
    base_iv_call_t = scenario_iv_from_curve(call_curve, S0, S_target, call_strike, dyn_mode)
    iv_call_t = crush_shifted_iv(base_iv_call_t if base_iv_call_t else call_iv_now, call_atm_now, scaled_call_atm_after)
    call_val_target = bs_price(S_target, call_strike, T_exit_call, r, iv_call_t/100.0, "C")

    # approximate decomp
    def _price_call(S, iv_pct):
        return bs_price(S, call_strike, T_exit_call, r, iv_pct/100.0, "C")
    vega_bump = max(1e-6, _price_call(S0, (scaled_call_atm_after + 1.0)) - _price_call(S0, scaled_call_atm_after))
    dS = S_target - S0
    dIV_pts = (scaled_call_atm_after - (call_atm_now if call_atm_now else call_iv_now))
    delta_gain = call_delta_now * dS
    call_val_now_exitIV = _price_call(S0, scaled_call_atm_after)
    theta_like = (call_val_target - call_val_now_exitIV) - delta_gain
    call_pnl_total = (call_val_target - call_cost_basis) * 100.0 * num_calls

    sanity = {
        "target_move_pct": target_move_pct,
        "adj_call_iv": scaled_call_atm_after,
        "call_delta_now": call_delta_now,
        "vega_bump": vega_bump,
        "delta_gain_total": delta_gain * 100.0 * num_calls,
        "vega_loss_total": vega_bump * dIV_pts * 100.0 * num_calls,
        "theta_like_total": theta_like * 100.0 * num_calls,
        "call_pnl_total": call_pnl_total
    }

    return prices, hedge_contracts, call_ret, hedge_ret, total_pl, sanity

# ---------- Run scenarios ----------
hedge_levels = [20, 50, 80, custom_hedge_ratio]
results = {}
sanity_ref = None

for ratio in hedge_levels:
    p, cts, cr, hr, tot, sanity = simulate_ratio_surface(ratio)
    results[ratio] = {"contracts": cts, "call": cr, "hedge": hr, "total": tot}
    if ratio == 50:
        sanity_ref = sanity

# ---------- Chart ----------
st.header("📊 P/L Comparison (vs Cost Basis)")
fig, ax = plt.subplots()
for ratio in hedge_levels:
    label = f"{ratio:.0f}% Hedge ({results[ratio]['contracts']} spd)" if ratio != custom_hedge_ratio else f"Custom {ratio:.0f}% ({results[ratio]['contracts']} spd)"
    ax.plot(p, results[ratio]["total"], label=label, linewidth=2)
ax.plot(p, results[50]["call"], "--", color="black", label="Unhedged Calls Only")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Stock Price at Exit")
ax.set_ylabel("Profit / Loss ($)")
ax.set_title(f"{symbol} — Projected Returns ({days_until_exit} days ahead) — Hedging {drop_pct}% over {risk_window_days}d ({dyn_mode})")
ax.legend()
st.pyplot(fig)

# ---------- Table with % Change ----------
st.header("📋 Projected Returns by Stock Price (vs Cost Basis)")
pct_change = ((p / S0) - 1.0) * 100.0
df = pd.DataFrame({"% Change": pct_change, "Stock Price": p, "Call Return": results[50]["call"]})
for ratio in hedge_levels:
    tag = f"{ratio:.0f}%"
    df[f"Hedge {tag} Return"] = results[ratio]["hedge"]
    df[f"Total P/L {tag}"]   = results[ratio]["total"]
st.dataframe(df.set_index(["% Change", "Stock Price"]).style.format("{:.0f}"))

# ---------- Suggested hedge (worst-case within your drop band) ----------
def worst_case(pl, prices, drop_pct, S0):
    band = prices <= (S0 * (1 - drop_pct / 100.0))
    if not np.any(band):
        return np.min(pl)
    return np.min(pl[band])

best_key = None
best_score = -1e18
for ratio in hedge_levels:
    score = worst_case(results[ratio]["total"], p, drop_pct, S0)
    if best_key is None or score > best_score:
        best_key, best_score = ratio, score

st.subheader("🧠 Suggested Hedge (focus: worst-case within your drop%)")
st.markdown(
    f"- **{best_key:.0f}% Hedge** → {results[best_key]['contracts']}× {symbol} {hedge_exp} "
    f"{long_strike:.0f}/{short_strike:.0f} Put Spread @ ~${spread_cost_live:.2f}/spread"
)

# ---------- Sanity panel for the call @ +0.8% ----------
if sanity_ref:
    st.subheader("🔍 Call P/L Sanity at +0.8% move (decomposition)")
    st.caption("Breaks your call’s next-day P/L into Delta gain, Vega (IV shift), and a Theta-like term.")
    st.write(
        f"- Scaled ATM Call IV after crush: **{sanity_ref['adj_call_iv']:.2f}%** "
    )
    st.write(
        f"- Δ (now) ≈ {sanity_ref['call_delta_now']:.2f} | "
        f"Vega bump ≈ ${sanity_ref['vega_bump']:.2f}/vol pt"
    )
    st.write(
        f"- Δ gain total ≈ ${sanity_ref['delta_gain_total']:,.0f} | "
        f"Vega total ≈ ${sanity_ref['vega_loss_total']:,.0f} | "
        f"θ-like ≈ ${sanity_ref['theta_like_total']:,.0f} | "
        f"**Call total ≈ ${sanity_ref['call_pnl_total']:,.0f}**"
    )

# ---------- Copyable Summary ----------
st.divider()
st.subheader("🧾 Copyable Results Summary")

summary_lines = [
    "=== OPTIONS HEDGE OPTIMIZER RESULTS (v10) ===",
    f"Symbol: {symbol}",
    f"Spot: {S0:.2f}",
    f"Your Calls: {num_calls}× {symbol} {call_exp} {call_strike}C @ ${call_cost_basis:.2f} (mid ~ ${call_mid:.2f}, IV ~ {call_iv_now:.1f}%)",
    f"Hedging Against: -{drop_pct}% over {risk_window_days} days  |  Expected Upside: +{expected_upside_pct}%",
    f"Surface Dynamics: {dyn_mode}",
    f"Hedge Structure: {symbol} {hedge_exp} {long_strike:.0f}/{short_strike:.0f} Put Spread (live cost ~ ${spread_cost_live:.2f}, put IV ~ {put_iv_now:.1f}%)",
    f"IV Crush (auto->call scaled ATM, skew preserved): {iv_crush:.1f}%",
    f"P/L Snapshot Horizon: {days_until_exit} days",
    "-"*44,
]
for ratio in hedge_levels:
    summary_lines.append(f"{ratio:.0f}% Hedge -> contracts {results[ratio]['contracts']}")
summary_lines.append("-"*44)
summary_lines.append("Table Columns: %Change | Stock Price | Call Return | Hedge Returns (20/50/80/Custom) | Total P/L (20/50/80/Custom)")

copy_block = "\n".join(summary_lines)
st.text_area("Copy these results:", value=copy_block, height=260)
st.download_button("Download Results (.txt)", data=copy_block, file_name=f"{symbol}_hedge_results_v10.txt")
