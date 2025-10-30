import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime, timedelta

st.set_page_config(page_title="Options Hedge Optimizer — Multi-Structure + Optimizer", layout="wide")
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
        return 1.0 if option_type == "C" and S > K else 0.0
    d1 = (log(S / K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1

# ---------- Header ----------
st.title("🧭 Options Hedge Optimizer — Multi-Structure, Auto IV Crush, Live Data + Optimizer")
st.markdown("""
Each hedge ratio (20%, 50%, 80%) uses a **different option structure**.  
Add your **Custom** ratio, and let the **Optimizer** suggest a hedge by either:
- **Minimizing Worst-Case Downside** (below current price), or
- **Maximizing Risk-Adjusted Return** (Sharpe-like) across the simulated price grid.

All returns shown are **vs your cost basis**.
""")

# ---------- Inputs ----------
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Stock Symbol", "MSFT").upper()
    call_strike = st.number_input("Call Strike", value=565.0)
    call_exp = st.text_input("Call Expiration (YYYY-MM-DD)", "2025-12-19")
    num_calls = st.number_input("Number of Call Contracts", value=3, step=1)
with col2:
    call_cost_basis = st.number_input("Avg Cost per Call ($)", value=13.49)
    days_until_exit = st.slider("Days Until Exit", 1, 30, 5)
    manual_iv_crush = st.slider("Manual IV Crush (%)", 0, 50, 15)
    custom_hedge_ratio = st.slider("Custom Hedge %", 0, 100, 65)

st.divider()

# ---------- Optimizer Controls ----------
opt_cols = st.columns(3)
with opt_cols[0]:
    objective = st.selectbox(
        "Optimizer Objective",
        ["Minimize Worst-Case Downside", "Maximize Risk-Adjusted Return"]
    )
with opt_cols[1]:
    sweep_enable = st.checkbox("Search 0–100% (step 5%) for optimal custom ratio", value=True)
with opt_cols[2]:
    downside_only_region = st.slider("Downside region width for 'worst-case' (as % below spot)", 5, 20, 10)

# ---------- Constants ----------
r = 0.05
today = datetime.today()

# ---------- Fetch spot (simple; Yahoo options are delayed anyway) ----------
stock = yf.Ticker(symbol)
hist = stock.history(period="1d")
if hist.empty:
    st.error("Could not fetch underlying price.")
    st.stop()
S0 = float(hist["Close"].iloc[-1])

# ---------- Pull option chain for your call ----------
try:
    chain_main = stock.option_chain(call_exp)
except Exception as e:
    st.error(f"Could not fetch options for {call_exp}: {e}")
    st.stop()

calls_df = chain_main.calls
call_row = calls_df[calls_df["strike"] == call_strike]
if call_row.empty:
    st.error(f"No {symbol} {call_strike} C for {call_exp}.")
    st.stop()

call_mid = float((call_row["bid"].iloc[0] + call_row["ask"].iloc[0]) / 2)
call_iv = float(call_row["impliedVolatility"].iloc[0] * 100)
st.caption(f"Using spot S0 = ${S0:.2f} | Your call mid ~ ${call_mid:.2f}, IV ~ {call_iv:.1f}%")

# ---------- Helper: pick expiry closest to target days ----------
def pick_exp(exp_list, target_days):
    parsed = []
    for d in exp_list:
        dt = datetime.strptime(d, "%Y-%m-%d")
        days = (dt - today).days
        if days > 0:
            parsed.append((d, days))
    if not parsed:
        return None
    return min(parsed, key=lambda x: abs(x[1] - target_days))[0]

# ---------- Hedge templates (independent structures) ----------
# 20%: cheaper tail hedge (deeper OTM, short)
# 50%: balanced (near OTM, weekly)
# 80%: deeper protection (closer to ATM, a bit longer)
hedge_templates = {
    20: {"otm_pct": 0.90, "days": 5,  "width_pct": 0.04},  # ~10% OTM, ~4% width
    50: {"otm_pct": 0.96, "days": 7,  "width_pct": 0.04},  # ~4% OTM, ~4% width
    80: {"otm_pct": 0.99, "days": 14, "width_pct": 0.03},  # ~1% OTM, ~3% width
}

# ---------- Build hedge structure for a ratio using its template ----------
def find_put_spread_for_ratio(ratio):
    tmpl = hedge_templates.get(ratio, hedge_templates[50])
    exp = pick_exp(stock.options, tmpl["days"])
    if exp is None:
        # fallback to nearest available
        exps = stock.options
        exp = exps[0] if exps else None
    if exp is None:
        st.error("No option expirations available.")
        st.stop()

    chain = stock.option_chain(exp)
    puts = chain.puts.dropna(subset=["strike", "bid", "ask", "impliedVolatility"])
    if puts.empty:
        st.error(f"No puts available for {exp}.")
        st.stop()

    long_target = S0 * tmpl["otm_pct"]
    width = S0 * tmpl["width_pct"]

    # choose long strike closest to target
    long_strike = float(puts.iloc[(puts["strike"] - long_target).abs().argsort()[:1]]["strike"])
    # choose short strike lower by ~width
    short_target = long_target - width
    short_strike = float(puts.iloc[(puts["strike"] - short_target).abs().argsort()[:1]]["strike"])

    row_long = puts[puts["strike"] == long_strike].iloc[0]
    row_short = puts[puts["strike"] == short_strike].iloc[0]

    long_mid = (row_long["bid"] + row_long["ask"]) / 2
    short_mid = (row_short["bid"] + row_short["ask"]) / 2
    spread_cost = float(long_mid - short_mid)
    put_iv = float(row_long["impliedVolatility"] * 100)

    return {
        "exp": exp,
        "up": long_strike,
        "low": short_strike,
        "iv": put_iv,
        "cost": spread_cost
    }

# ---------- Simulation per ratio & structure ----------
def simulate_ratio(hedge_ratio, exp, up_strike, low_strike, iv_put, cost_basis):
    # Size hedge contracts to target delta coverage
    T_call = (datetime.strptime(call_exp, "%Y-%m-%d") - today).days / 365
    delta_call = bs_delta(S0, call_strike, T_call, call_iv / 100, "C")
    total_call_delta = delta_call * num_calls * 100

    T_put = (datetime.strptime(exp, "%Y-%m-%d") - today).days / 365
    delta_put = abs(bs_delta(S0, up_strike, T_put, iv_put / 100, "P")) * 100
    hedge_contracts = max(1, round((total_call_delta * (hedge_ratio / 100)) / delta_put))

    # Price grid
    prices = np.arange(S0 * 0.9, S0 * 1.1, 2.0)

    # Exit-time vols / times
    T_exit_call = (datetime.strptime(call_exp, "%Y-%m-%d") - today - timedelta(days=days_until_exit)).days / 365
    iv_crush = manual_iv_crush  # (keep manual; swap for auto if you add that back)
    iv_call_new = call_iv * (1 - iv_crush / 100)
    iv_put_new = iv_put * (1 - iv_crush / 100)

    def val(S, K, T, sigma, opt_type):
        return bs_price(S, K, T, r, sigma / 100, opt_type)

    call_val = np.array([val(p, call_strike, T_exit_call, iv_call_new, "C") for p in prices])
    put_up = np.array([val(p, up_strike, T_put - days_until_exit / 365, iv_put_new, "P") for p in prices])
    put_lo = np.array([val(p, low_strike, T_put - days_until_exit / 365, iv_put_new, "P") for p in prices])

    call_ret = (call_val - call_cost_basis) * num_calls * 100
    hedge_ret = hedge_contracts * ((put_up - put_lo) - cost_basis) * 100
    total = call_ret + hedge_ret

    return prices, hedge_contracts, call_ret, hedge_ret, total

# ---------- Compute independent structures for 20/50/80 ----------
hedge_levels = [20, 50, 80]
structures = {ratio: find_put_spread_for_ratio(ratio) for ratio in hedge_levels}

results = {}
for ratio in hedge_levels:
    s = structures[ratio]
    p, c, cr, hr, t = simulate_ratio(ratio, s["exp"], s["up"], s["low"], s["iv"], s["cost"])
    results[ratio] = {"contracts": c, "call": cr, "hedge": hr, "total": t, **s}

# ---------- Custom hedge uses the 50%-style structure by default ----------
custom_structure = find_put_spread_for_ratio(50)
p, c, cr, hr, t = simulate_ratio(custom_hedge_ratio, custom_structure["exp"], custom_structure["up"],
                                 custom_structure["low"], custom_structure["iv"], custom_structure["cost"])
results["custom"] = {"contracts": c, "call": cr, "hedge": hr, "total": t, **custom_structure}

# ---------- Optimizer metrics ----------
def sharpe_like(pl):
    mu = np.mean(pl)
    sd = np.std(pl)
    return mu / sd if sd > 1e-9 else np.nan

def worst_case_downside(pl, prices, spot):
    mask = prices <= spot
    if not np.any(mask):
        return np.nan
    return np.min(pl[mask])

# Evaluate discrete candidates (20/50/80/custom)
scores = []
for key, val in results.items():
    tag = f"{key}%" if key != "custom" else f"Custom {custom_hedge_ratio:.0f}%"
    total = val["total"]
    if objective == "Maximize Risk-Adjusted Return":
        score = sharpe_like(total)
        better_higher = True
    else:
        score = worst_case_downside(total, p, S0 * (1 - downside_only_region / 100))  # worst within downside band
        better_higher = True  # less negative (higher) is better

    scores.append({"Scenario": tag, "Score": score, "Contracts": val["contracts"],
                   "Exp": val["exp"], "Strikes": f"{val['up']:.0f}/{val['low']:.0f}", "Cost/Spread": val["cost"]})

# Pick best among discrete candidates
scores_df = pd.DataFrame(scores)
if objective == "Maximize Risk-Adjusted Return":
    best_idx = scores_df["Score"].idxmax()
else:
    best_idx = scores_df["Score"].idxmax()
best_discrete = scores_df.loc[best_idx]

# Optional: sweep 0–100% for "optimal" custom ratio using the balanced structure
sweep_best = None
sweep_table = None
if sweep_enable:
    sweep_rows = []
    for ratio in range(0, 101, 5):
        p_s, c_s, cr_s, hr_s, t_s = simulate_ratio(
            ratio,
            custom_structure["exp"], custom_structure["up"],
            custom_structure["low"], custom_structure["iv"], custom_structure["cost"]
        )
        if objective == "Maximize Risk-Adjusted Return":
            score = sharpe_like(t_s)
        else:
            score = worst_case_downside(t_s, p_s, S0 * (1 - downside_only_region / 100))
        sweep_rows.append({"Ratio%": ratio, "Score": score, "Contracts": c_s})
    sweep_table = pd.DataFrame(sweep_rows)
    # choose best according to objective
    sweep_best = sweep_table.iloc[sweep_table["Score"].idxmax()]

# ---------- Chart ----------
st.header("📊 P/L Comparison Across Hedge Structures")
fig, ax = plt.subplots()
palette = {20: None, 50: None, 80: None, "custom": None}
for key, val in results.items():
    label = f"{key}% Hedge ({val['contracts']} spd)" if key != "custom" else f"Custom {custom_hedge_ratio:.0f}% ({val['contracts']} spd)"
    ax.plot(p, val["total"], label=label, linewidth=2)
ax.plot(p, results[50]["call"], "--", color="black", label="Unhedged Calls Only")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Stock Price at Exit")
ax.set_ylabel("Profit / Loss ($)")
ax.set_title(f"{symbol} - Projected Returns ({days_until_exit} days ahead)")
ax.legend()
st.pyplot(fig)

# ---------- P&L Table ----------
st.header("📋 Projected Returns by Stock Price (vs Cost Basis)")
df = pd.DataFrame({"Stock Price": p, "Call Return": results[50]["call"]})
for key, val in results.items():
    tag = f"{key}%" if key != "custom" else f"Custom {custom_hedge_ratio:.0f}%"
    df[f"Hedge {tag} Return"] = val["hedge"]
    df[f"Total P/L {tag}"] = val["total"]
st.dataframe(df.set_index("Stock Price").style.format("{:.0f}"))

# ---------- Recommendation Panel ----------
st.subheader("🧠 Optimal Hedge — Recommendation")
colA, colB = st.columns(2)

with colA:
    st.markdown("**Best among 20% / 50% / 80% / Custom**")
    st.write(f"- **Objective:** {objective}")
    st.write(f"- **Recommended:** {best_discrete['Scenario']}")
    st.write(f"- **Score:** {best_discrete['Score']:.4f}")
    st.write(f"- **Contracts:** {int(best_discrete['Contracts'])}")
    st.write(f"- **Structure:** {symbol} {best_discrete['Exp']} {best_discrete['Strikes']} Put Spread")
    st.write(f"- **Cost/Spread:** ${float(best_discrete['Cost/Spread']):.2f}")

with colB:
    if sweep_enable and sweep_best is not None:
        st.markdown("**Best Custom Ratio (0–100%, balanced structure)**")
        st.write(f"- **Objective:** {objective}")
        st.write(f"- **Optimal Ratio:** {int(sweep_best['Ratio%'])}%")
        st.write(f"- **Score:** {sweep_best['Score']:.4f}")
        st.write(f"- **Contracts:** {int(sweep_best['Contracts'])}")
        st.caption("Custom sweep uses the balanced (50%) expiry/strikes; only the ratio (and thus contracts) is varied.")

# ---------- Details for each hedge ----------
st.subheader("🧾 Hedge Position Details")
for key, val in results.items():
    tag = f"{key}%" if key != "custom" else f"Custom {custom_hedge_ratio:.0f}%"
    total_cost = val["contracts"] * val["cost"] * 100
    st.markdown(
        f"- **{tag} Hedge:** {val['contracts']}× {symbol} {val['exp']} "
        f"{val['up']:.0f}/{val['low']:.0f} Put Spread | Cost ${val['cost']:.2f}/spread → **Total ≈ ${total_cost:.0f}**"
    )

# ---------- Optional: show optimizer sweep table ----------
if sweep_enable and sweep_table is not None:
    st.markdown("#### Optimizer Sweep (balanced structure)")
    st.dataframe(sweep_table.set_index("Ratio%").style.format("{:.4f}", subset=["Score"]))

# ---------- Copyable Summary Output ----------
st.divider()
st.subheader("🧾 Copyable Results Summary")

summary_lines = [
    "=== OPTIONS HEDGE OPTIMIZER RESULTS ===",
    f"Symbol: {symbol}",
    f"Spot: {S0:.2f}",
    f"Call Strike: {call_strike}  | Exp: {call_exp}  | Contracts: {num_calls}  | Cost Basis: ${call_cost_basis}",
    f"Days Until Exit: {days_until_exit}",
    f"IV Crush: {manual_iv_crush}%",
    f"Custom Hedge: {custom_hedge_ratio}%",
    f"Objective: {objective}",
    "-"*40,
    f"Best Discrete Hedge: {best_discrete['Scenario']} ({symbol} {best_discrete['Exp']} {best_discrete['Strikes']} Put Spread)",
    f"Contracts: {int(best_discrete['Contracts'])} | Cost/Spread: ${float(best_discrete['Cost/Spread']):.2f} | Score: {best_discrete['Score']:.4f}",
]

if sweep_enable and sweep_best is not None:
    summary_lines += [
        f"Optimal Custom Ratio (sweep): {int(sweep_best['Ratio%'])}%",
        f"Sweep Score: {sweep_best['Score']:.4f}",
    ]

summary_lines.append("-"*40)
summary_lines.append("Hedge Structures:")
for key, val in results.items():
    tag = f"{key}%" if key != "custom" else f"Custom {custom_hedge_ratio:.0f}%"
    summary_lines.append(
        f"  {tag} -> {val['up']:.0f}/{val['low']:.0f} exp {val['exp']} | cost ${val['cost']:.2f} | contracts {val['contracts']}"
    )

copy_block = "\n".join(summary_lines)

st.text_area(
    "Full Text Summary (copy below to paste into ChatGPT or your notes):",
    value=copy_block,
    height=260,
)
