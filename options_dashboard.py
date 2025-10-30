import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Options Dashboard", layout="wide")

st.title("📊 Stock Options Dashboard")
st.write("View option chain volume, open interest, and premiums — powered by Yahoo Finance")

# ---------- Helpers ----------
CALLS_GREEN = "#2ECC40"
PUTS_RED = "#FF4136"
DARK_CALLS = "#1E8449"
DARK_PUTS = "#922B21"


def fmt_date_str(d):
    try:
        return pd.to_datetime(d).strftime("%b %d %Y")
    except Exception:
        return str(d)


def dominant_series(call_s: pd.Series, put_s: pd.Series) -> pd.Series:
    total = call_s.fillna(0) + put_s.fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_calls = np.where(total == 0, np.nan, (call_s / total) * 100.0)
    out = []
    for x in pct_calls:
        if np.isnan(x):
            out.append("—")
        else:
            out.append(f"{x:.0f}% Calls" if x >= 50 else f"{100-x:.0f}% Puts")
    return pd.Series(out, index=call_s.index)


def build_summary_df(summary_df: pd.DataFrame, expirations: list) -> pd.DataFrame:
    wide = summary_df.pivot(
        index="expiration",
        columns="type",
        values=["volume", "openInterest", "premium_volume", "premium_oi"],
    )
    wide.columns = [f"{m}_{t}" for m, t in wide.columns]
    wide = wide.reindex(expirations)

    wide["dom_vol"] = dominant_series(wide["volume_call"], wide["volume_put"])
    wide["dom_oi"] = dominant_series(wide["openInterest_call"], wide["openInterest_put"])
    wide["dom_pv"] = dominant_series(wide["premium_volume_call"], wide["premium_volume_put"])
    wide["dom_poi"] = dominant_series(wide["premium_oi_call"], wide["premium_oi_put"])

    cols = [
        ("", "Expiration"),
        ("Contracts", "Volume Call"),
        ("Contracts", "Volume Put"),
        ("Contracts", "Volume Majority"),
        ("Contracts", "OI Call"),
        ("Contracts", "OI Put"),
        ("Contracts", "OI Majority"),
        ("Premiums", "Prem Vol Call"),
        ("Premiums", "Prem Vol Put"),
        ("Premiums", "Prem Vol Majority"),
        ("Premiums", "Prem OI Call"),
        ("Premiums", "Prem OI Put"),
        ("Premiums", "Prem OI Majority"),
    ]

    out = pd.DataFrame(index=wide.index, columns=pd.MultiIndex.from_tuples(cols))
    out[("", "Expiration")] = [fmt_date_str(x) for x in wide.index]

    out[("Contracts", "Volume Call")] = wide["volume_call"].astype("Int64")
    out[("Contracts", "Volume Put")] = wide["volume_put"].astype("Int64")
    out[("Contracts", "Volume Majority")] = wide["dom_vol"]
    out[("Contracts", "OI Call")] = wide["openInterest_call"].astype("Int64")
    out[("Contracts", "OI Put")] = wide["openInterest_put"].astype("Int64")
    out[("Contracts", "OI Majority")] = wide["dom_oi"]
    out[("Premiums", "Prem Vol Call")] = wide["premium_volume_call"].astype("Int64")
    out[("Premiums", "Prem Vol Put")] = wide["premium_volume_put"].astype("Int64")
    out[("Premiums", "Prem Vol Majority")] = wide["dom_pv"]
    out[("Premiums", "Prem OI Call")] = wide["premium_oi_call"].astype("Int64")
    out[("Premiums", "Prem OI Put")] = wide["premium_oi_put"].astype("Int64")
    out[("Premiums", "Prem OI Majority")] = wide["dom_poi"]

    return out.reset_index(drop=True)


def build_totals_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    sums = summary_df.groupby("type")[["volume", "openInterest", "premium_volume", "premium_oi"]].sum()

    def dom(c, p):
        total = c + p
        if total == 0:
            return "—"
        x = (c / total) * 100.0
        return f"{x:.0f}% Calls" if x >= 50 else f"{100-x:.0f}% Puts"

    row = {
        ("", "Expiration"): "All Expirations",
        ("Contracts", "Volume Call"): int(sums.loc["call", "volume"]),
        ("Contracts", "Volume Put"): int(sums.loc["put", "volume"]),
        ("Contracts", "Volume Majority"): dom(int(sums.loc["call", "volume"]), int(sums.loc["put", "volume"])),
        ("Contracts", "OI Call"): int(sums.loc["call", "openInterest"]),
        ("Contracts", "OI Put"): int(sums.loc["put", "openInterest"]),
        ("Contracts", "OI Majority"): dom(
            int(sums.loc["call", "openInterest"]), int(sums.loc["put", "openInterest"])
        ),
        ("Premiums", "Prem Vol Call"): int(sums.loc["call", "premium_volume"]),
        ("Premiums", "Prem Vol Put"): int(sums.loc["put", "premium_volume"]),
        ("Premiums", "Prem Vol Majority"): dom(
            int(sums.loc["call", "premium_volume"]), int(sums.loc["put", "premium_volume"])
        ),
        ("Premiums", "Prem OI Call"): int(sums.loc["call", "premium_oi"]),
        ("Premiums", "Prem OI Put"): int(sums.loc["put", "premium_oi"]),
        ("Premiums", "Prem OI Majority"): dom(
            int(sums.loc["call", "premium_oi"]), int(sums.loc["put", "premium_oi"])
        ),
    }
    df = pd.DataFrame([row])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def style_tables(df: pd.DataFrame):
    def color_dom(s: pd.Series):
        styles = []
        for v in s:
            if isinstance(v, str) and "Calls" in v:
                styles.append(f"color: {CALLS_GREEN}; font-weight: 600;")
            elif isinstance(v, str) and "Puts" in v:
                styles.append(f"color: {PUTS_RED}; font-weight: 600;")
            else:
                styles.append("")
        return styles

    def alt_rows(row):
        # transparent (none) on even rows, lightly shaded on odd rows
        return [
            "background-color: rgba(0, 0, 0, 0.0)" if row.name % 2 == 0
            else "background-color: rgba(0, 0, 0, 0.05)"  # 90% transparent
            for _ in row
        ]

    styler = df.style.format(
        lambda x: ""
        if pd.isna(x)
        else f"${int(x):,}" if isinstance(x, (int, np.integer)) and "Prem" in str(df.columns)
        else f"{int(x):,}" if isinstance(x, (int, np.integer)) else x
    )

    for col in [
        ("Contracts", "Volume Majority"),
        ("Contracts", "OI Majority"),
        ("Premiums", "Prem Vol Majority"),
        ("Premiums", "Prem OI Majority"),
    ]:
        if col in df.columns:
            styler = styler.apply(color_dom, subset=[col])

    styler = (
        styler.set_table_styles(
            [
                {
                    "selector": "th.col_heading.level0",
                    "props": [("text-align", "center"), ("font-weight", "700"), ("border-bottom", "2px solid #bbb")],
                },
                {
                    "selector": "th.col_heading.level1",
                    "props": [("text-align", "center"), ("font-weight", "600")],
                },
                {"selector": "td", "props": [("padding", "6px 10px"), ("border-bottom", "1px solid #eee")]},
            ]
        )
        .apply(alt_rows, axis=1)
        .hide(axis="index")
    )

    return styler


# ---------- Input ----------
symbol = st.text_input("Enter stock symbol (e.g. AAPL, SPY, TSLA):", "AAPL").upper()

if symbol:
    try:
        ticker = yf.Ticker(symbol)
        expirations = list(ticker.options or [])

        if not expirations:
            st.error("No options data found for this symbol.")
        else:
            rows, all_data = [], []

            with st.spinner(f"Fetching all expiration data for {symbol}..."):
                for exp in expirations:
                    chain = ticker.option_chain(exp)
                    calls, puts = chain.calls.copy(), chain.puts.copy()
                    calls["type"], puts["type"] = "call", "put"
                    df = pd.concat([calls, puts], ignore_index=True)
                    df["expiration"] = exp
                    df["premium_volume"] = df["lastPrice"].fillna(0) * df["volume"].fillna(0) * 100
                    df["premium_oi"] = df["lastPrice"].fillna(0) * df["openInterest"].fillna(0) * 100
                    all_data.append(df)

                    summary = (
                        df.groupby("type")[["volume", "openInterest", "premium_volume", "premium_oi"]]
                        .sum()
                        .reset_index()
                    )
                    summary["expiration"] = exp
                    rows.append(summary)

            summary_df = pd.concat(rows, ignore_index=True)
            full_df = pd.concat(all_data, ignore_index=True)
            for col in ["volume", "openInterest", "premium_volume", "premium_oi"]:
                summary_df[col] = summary_df[col].fillna(0).astype(int)

            # ---------- Charts ----------
            calls_df = summary_df[summary_df["type"] == "call"].set_index("expiration").reindex(expirations).fillna(0)
            puts_df = summary_df[summary_df["type"] == "put"].set_index("expiration").reindex(expirations).fillna(0)
            exp_labels = [str(e) for e in expirations]

            def add_vlines(fig, x_labels):
                for label in x_labels[:-1]:
                    fig.add_vline(x=label, line=dict(color="lightgray", width=1, dash="dash"), opacity=0.5)

            # --- Chart 1 toggles ---
            show_vol = st.checkbox("Show Volume", value=True, key="vol")
            show_oi = st.checkbox("Show Open Interest", value=True, key="oi")

            fig1 = go.Figure()
            if show_vol:
                fig1.add_trace(go.Bar(x=exp_labels, y=calls_df["volume"], name="Calls Volume", marker_color=CALLS_GREEN, offsetgroup="VOL"))
                fig1.add_trace(go.Bar(x=exp_labels, y=puts_df["volume"], name="Puts Volume", marker_color=PUTS_RED, offsetgroup="VOL"))
            if show_oi:
                fig1.add_trace(go.Bar(x=exp_labels, y=calls_df["openInterest"], name="Calls OI", marker_color=DARK_CALLS, offsetgroup="OI"))
                fig1.add_trace(go.Bar(x=exp_labels, y=puts_df["openInterest"], name="Puts OI", marker_color=DARK_PUTS, offsetgroup="OI"))

            add_vlines(fig1, exp_labels)
            fig1.update_layout(
                barmode="stack",
                bargap=0.35,
                title=f"{symbol} — Volume and Open Interest by Expiration",
                xaxis_title="Expiration",
                yaxis_title="Contracts",
                xaxis_tickangle=-45,
                height=550,
                xaxis=dict(
                    type="category",
                    categoryorder="array",
                    categoryarray=exp_labels,
                    tickmode="array",
                    tickvals=exp_labels,
                    ticktext=[fmt_date_str(e) for e in expirations],
                ),
            )
            st.plotly_chart(fig1, use_container_width=True)

            # --- Chart 2 toggles ---
            show_prem_vol = st.checkbox("Show Premium Volume", value=True, key="prem_vol")
            show_prem_oi = st.checkbox("Show Premium OI", value=True, key="prem_oi")

            fig2 = go.Figure()
            if show_prem_vol:
                fig2.add_trace(go.Bar(x=exp_labels, y=calls_df["premium_volume"], name="Calls Premium (Vol)", marker_color=CALLS_GREEN, offsetgroup="PREM_VOL"))
                fig2.add_trace(go.Bar(x=exp_labels, y=puts_df["premium_volume"], name="Puts Premium (Vol)", marker_color=PUTS_RED, offsetgroup="PREM_VOL"))
            if show_prem_oi:
                fig2.add_trace(go.Bar(x=exp_labels, y=calls_df["premium_oi"], name="Calls Premium (OI)", marker_color=DARK_CALLS, offsetgroup="PREM_OI"))
                fig2.add_trace(go.Bar(x=exp_labels, y=puts_df["premium_oi"], name="Puts Premium (OI)", marker_color=DARK_PUTS, offsetgroup="PREM_OI"))

            add_vlines(fig2, exp_labels)
            fig2.update_layout(
                barmode="stack",
                bargap=0.35,
                title=f"{symbol} — Total Premium Value by Expiration",
                xaxis_title="Expiration",
                yaxis_title="Total Premium ($)",
                xaxis_tickangle=-45,
                height=550,
                xaxis=dict(
                    type="category",
                    categoryorder="array",
                    categoryarray=exp_labels,
                    tickmode="array",
                    tickvals=exp_labels,
                    ticktext=[fmt_date_str(e) for e in expirations],
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ---------- Tables ----------
            st.subheader(f"📅 Summary for {symbol} — All Expirations")
            st.dataframe(style_tables(build_summary_df(summary_df, expirations)), use_container_width=True)

            st.subheader("📈 Overall Totals Across All Expirations")
            st.dataframe(style_tables(build_totals_df(summary_df)), use_container_width=True)

            # ---------- Top 10 ----------
            st.markdown("---")
            st.header("🏆 Top 10 Option Contracts")

            def top10(df, metric):
                cols = ["expiration", "strike", "type", "lastPrice", metric]
                sub = df[cols].copy().sort_values(metric, ascending=False).head(10)
                sub["expiration"] = sub["expiration"].apply(fmt_date_str)
                if "premium" in metric:
                    sub[metric] = sub[metric].apply(lambda x: f"${x:,.0f}")
                else:
                    sub[metric] = sub[metric].apply(lambda x: f"{x:,}")
                return sub.rename(
                    columns={
                        "expiration": "Expiration",
                        "strike": "Strike",
                        "type": "Type",
                        "lastPrice": "Last Price",
                        metric: metric.replace("_", " ").title(),
                    }
                )

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Top 10 by Volume")
                st.dataframe(top10(full_df, "volume"), use_container_width=True)
            with c2:
                st.subheader("Top 10 by Open Interest")
                st.dataframe(top10(full_df, "openInterest"), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Top 10 by Premium Volume ($)")
                st.dataframe(top10(full_df, "premium_volume"), use_container_width=True)
            with c4:
                st.subheader("Top 10 by Premium OI ($)")
                st.dataframe(top10(full_df, "premium_oi"), use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
