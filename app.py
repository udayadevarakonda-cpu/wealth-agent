import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from screener import (
    get_nifty_macro_regime,
    scan_strong_buys_with_allocation,
    analyze_single_asset,
    scan_core_mutual_funds,
    get_universe_tickers,
    run_scorecard_scan,
    run_historical_backtest,
    get_live_ipo_calendar,
    build_ipo_recommendation,
    DEFAULT_SCORECARD_WEIGHTS
)

st.set_page_config(page_title="Strategic Wealth & Tactical Engine", layout="wide", page_icon="📈")
st.title("🛡️ Strategic Wealth & Tactical Satellite Engine")

# =============================================================================
# REFRESH CONTROLS & CACHE MANAGEMENT
# =============================================================================
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

if "universe_bust" not in st.session_state:
    st.session_state.universe_bust = 0

if "scorecard_bust" not in st.session_state:
    st.session_state.scorecard_bust = 0


@st.cache_data(ttl=900, show_spinner=False)
def cached_macro_regime(bust):
    return get_nifty_macro_regime()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_universe(index_name, universe_bust):
    return get_universe_tickers(index_name)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_scorecard_scan(tickers, budget, top_n, weights_tuple, liquidity_floor, max_leverage, bust):
    weights = dict(weights_tuple)
    return run_scorecard_scan(
        tickers, satellite_budget=budget, top_n=top_n, weights=weights,
        liquidity_floor_inr=liquidity_floor, max_leverage_multiple=max_leverage,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_stock_scan(tickers, budget, bust):
    return scan_strong_buys_with_allocation(tickers, satellite_budget=budget)

@st.cache_data(ttl=900, show_spinner=False)
def cached_ipo_scan(bust):
    return get_live_ipo_calendar()


@st.cache_data(ttl=900, show_spinner=False)
def cached_mf_scan(bust):
    return scan_core_mutual_funds()


@st.cache_data(ttl=900, show_spinner=False)
def cached_single_asset(query, bust):
    return analyze_single_asset(query)


def render_skip_report(meta, label="items"):
    attempted = meta.get("attempted", 0)
    succeeded = meta.get("succeeded", 0)
    skipped = meta.get("skipped", [])
    st.caption(f"Data pull: {succeeded}/{attempted} {label} loaded successfully.")
    if skipped:
        with st.expander(f"⚠️ {len(skipped)} {label} skipped — click to inspect"):
            for name, reason in skipped:
                st.write(f"- **{name}**: {reason}")


# =============================================================================
# SIDEBAR: FINANCIAL BASELINE & ALLOCATION SLICERS
# =============================================================================
st.sidebar.header("⚙️ Portfolio & Runway Slicers")

with st.sidebar.expander("1️⃣ Financial Baseline & Runway", expanded=True):
    total_managed_capital = st.number_input(
        "Total Liquid Capital (₹)",
        min_value=100000.0,
        max_value=20000000.0,
        value=1147000.0,
        step=25000.0
    )
    monthly_expense = st.number_input(
        "Monthly Household Spend (₹)",
        min_value=20000.0,
        max_value=500000.0,
        value=100000.0,
        step=5000.0
    )
    emergency_months = st.slider(
        "Emergency Runway Buffer (Months)",
        min_value=3,
        max_value=12,
        value=4,
        step=1,
        help="Allocated directly to Liquid / Overnight Funds for immediate liquidity."
    )

emergency_liquid_alloc = min(monthly_expense * emergency_months, total_managed_capital)

with st.sidebar.expander("2️⃣ Preservation & Satellite Split", expanded=True):
    tactical_satellite_pct = st.slider(
        "Tactical Satellite Bucket (%)",
        min_value=1.0,
        max_value=10.0,
        value=2.4,
        step=0.1,
        help="High-velocity momentum stock allocation with strict dynamic ATR stops."
    )
    arbitrage_ratio_pct = st.slider(
        "Intermediate Yield: Arbitrage vs Debt (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        help="Split of remaining capital between Arbitrage (tax-efficient) and Short-Term Debt."
    )
    macro_alert_threshold = st.sidebar.slider(
        "Breakout Proximity Alert (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.25,
        help="Triggers an early-warning alert when Nifty approaches within this % of its 200 EMA."
    )

with st.sidebar.expander("3️⃣ Core Equity Phase-in (SIP)", expanded=True):
    weekly_sip_amount = st.number_input(
        "Weekly Equity Phase-in SIP (₹/week)",
        min_value=1000.0,
        max_value=50000.0,
        value=7500.0,
        step=500.0
    )
    large_cap_pct = st.slider("Large Cap / Index Split (%)", min_value=10, max_value=80, value=50, step=5)
    flexi_cap_pct = st.slider("Flexi Cap Split (%)", min_value=10, max_value=80, value=35, step=5)
    mid_small_pct = max(0, 100 - (large_cap_pct + flexi_cap_pct))
    st.info(f"📊 Mid/Small Cap Split (Auto-balanced): **{mid_small_pct}%**")

with st.sidebar.expander("4️⃣ Tactical Scan Universe", expanded=False):
    universe_choice = st.selectbox(
        "Scan Universe (live NSE index constituents)",
        ["NIFTY 50", "NIFTY 100", "NIFTY 200"],
        index=1,
        help="Constituents loaded directly from NSE's archive CSVs."
    )
    if st.button("🔄 Refresh Universe List", use_container_width=True):
        st.session_state.universe_bust += 1
        st.rerun()

with st.sidebar.expander("5️⃣ Scorecard: Weights & Vetoes", expanded=False):
    top_n_stocks = st.slider("Show Top N by Composite Score", min_value=5, max_value=30, value=15, step=1)
    st.markdown("**Bucket Weights** (auto-normalized to 100%)")
    w_momentum = st.slider("Momentum (SuperTrend/ADX/EMA)", 0, 100, 40, 5)
    w_quality = st.slider("Quality (ROIC/Interest Coverage/Leverage/Growth)", 0, 100, 30, 5)
    w_valuation = st.slider("Valuation (PE/P-B/Dividend Yield, sector-relative)", 0, 100, 20, 5)
    w_liquidity = st.slider("Liquidity (Avg Daily Traded Value)", 0, 100, 10, 5)
    _w_total = max(1, w_momentum + w_quality + w_valuation + w_liquidity)
    scorecard_weights = {
        "momentum": w_momentum / _w_total, "quality": w_quality / _w_total,
        "valuation": w_valuation / _w_total, "liquidity": w_liquidity / _w_total,
    }
    st.caption(f"Normalized: Momentum {scorecard_weights['momentum']:.0%} · Quality {scorecard_weights['quality']:.0%} · "
               f"Valuation {scorecard_weights['valuation']:.0%} · Liquidity {scorecard_weights['liquidity']:.0%}")

    st.markdown("**Hard Vetoes**")
    liquidity_floor_cr = st.number_input("Min Avg Daily Traded Value (₹ Crore)", min_value=0.1, max_value=100.0, value=1.0, step=0.5)
    max_leverage_multiple = st.number_input("Max Debt/Equity vs Sector Median (multiple)", min_value=1.5, max_value=10.0, value=3.0, step=0.5)

    if st.button("🔄 Refresh Scorecard Data", use_container_width=True):
        st.session_state.scorecard_bust += 1
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Full Data Refresh", use_container_width=True):
    st.session_state.refresh_counter += 1
    st.cache_data.clear()
    st.rerun()

# Mathematical Allocations
tactical_satellite_alloc = round(total_managed_capital * (tactical_satellite_pct / 100.0), 2)
remaining_capital = max(0.0, total_managed_capital - emergency_liquid_alloc - tactical_satellite_alloc)

arb_alloc = round(remaining_capital * (arbitrage_ratio_pct / 100.0), 2)
debt_alloc = round(remaining_capital - arb_alloc, 2)

core_total = emergency_liquid_alloc + arb_alloc + debt_alloc
core_pct = (core_total / total_managed_capital) * 100 if total_managed_capital > 0 else 0
sat_pct = (tactical_satellite_alloc / total_managed_capital) * 100 if total_managed_capital > 0 else 0

sip_large = weekly_sip_amount * (large_cap_pct / 100.0)
sip_flexi = weekly_sip_amount * (flexi_cap_pct / 100.0)
sip_mid_small = weekly_sip_amount * (mid_small_pct / 100.0)

# =============================================================================
# REAL-TIME MACRO REGIME & PROXIMITY BANNER
# =============================================================================
macro_status = cached_macro_regime(st.session_state.refresh_counter)
macro_data_ok = bool(macro_status.get("ok"))

if not macro_data_ok:
    st.error(
        f"⚠️ **MACRO DATA UNAVAILABLE**\n\n"
        f"Nifty 50 / 200-EMA could not be fetched (`{macro_status.get('error', 'unknown error')}`). "
        f"Tactical Satellite entries are held on standby."
    )
    n_price = n_ema = diff_pts = diff_pct = None
    is_bull = False
else:
    n_price = macro_status["nifty_price"]
    n_ema = macro_status["ema_200"]
    is_bull = macro_status["bullish"]
    diff_pts = macro_status["diff_pts"]
    diff_pct = macro_status["diff_pct"]
    as_of = macro_status["as_of"]

    if is_bull:
        st.success(
            f"🟢 **MACRO GATE IS BULLISH:** Nifty 50 (₹{n_price:,.2f}) is trading "
            f"**+{diff_pct}% (+{diff_pts} pts)** above its 200-Day EMA (₹{n_ema:,.2f}). *(as of {as_of})*\n\n"
            f"👉 **Execution Clear:** Tactical Satellite capital (₹{tactical_satellite_alloc:,.2f}) is cleared to deploy into qualified momentum stocks in Tab 2."
        )
    elif abs(diff_pct) <= macro_alert_threshold:
        st.warning(
            f"⚡ **MACRO ALERT — IMMINENT BREAKOUT PROXIMITY:** Nifty 50 (₹{n_price:,.2f}) is only "
            f"**{abs(diff_pct)}% ({abs(diff_pts)} pts)** away from crossing above its 200-Day EMA (₹{n_ema:,.2f}). *(as of {as_of})*\n\n"
            f"👉 Prepare tactical order tickets. Keep ₹{tactical_satellite_alloc:,.2f} ready in ICICI Arbitrage for fast deployment upon crossover."
        )
    else:
        st.error(
            f"🔴 **MACRO GATE IS BEARISH / CAUTION:** Nifty 50 (₹{n_price:,.2f}) is "
            f"**{diff_pct}% ({diff_pts} pts)** below its 200-Day EMA (₹{n_ema:,.2f}). *(as of {as_of})*\n\n"
            f"🛡️ **Capital Protected:** Tactical satellite reserve of **₹{tactical_satellite_alloc:,.2f}** remains parked in **ICICI Prudential Arbitrage Fund** earning ~7% yield with zero equity drawdown risk."
        )

st.markdown("---")

# =============================================================================
# MAIN TABS (4 TABS)
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Strategic Asset Allocation & Detailed Blueprint",
    "🎯 Tactical Stock Momentum",
    "📈 Core Mutual Fund Qualifier Engine",
    "🧪 Scenario Sandbox & Backtest Simulator"
])

# =============================================================================
# TAB 1: BLUEPRINT & RUNWAY ALLOCATION
# =============================================================================
with tab1:
    st.subheader("📊 Capital Deployment Framework & Strategic Blueprint")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Liquid Capital", f"₹{total_managed_capital:,.2f}")
    c2.metric("Emergency Runway", f"{emergency_months} Months", f"₹{emergency_liquid_alloc:,.2f} Liquid")
    c3.metric("Core Preservation", f"₹{core_total:,.2f}", f"{core_pct:.1f}%")
    c4.metric("Tactical Satellite", f"₹{tactical_satellite_alloc:,.2f}", f"{sat_pct:.1f}%")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("#### 🏛️ Portfolio Capital Distribution")
        alloc_data = pd.DataFrame([
            {"Bucket": f"Liquid Buffer ({emergency_months}M)", "Amount (₹)": emergency_liquid_alloc},
            {"Bucket": "Arbitrage Fund", "Amount (₹)": arb_alloc},
            {"Bucket": "Short-Term Debt", "Amount (₹)": debt_alloc},
            {"Bucket": "Tactical Satellite", "Amount (₹)": tactical_satellite_alloc}
        ])
        fig_pie = px.pie(alloc_data, names="Bucket", values="Amount (₹)", hole=0.45, color_discrete_sequence=px.colors.qualitative.Safe)
        fig_pie.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("#### 🎯 Target Capital Parking Schedule")
        parking_df = pd.DataFrame([
            {"Bucket": "Emergency Runway", "Allocated (₹)": f"₹{emergency_liquid_alloc:,.2f}", "Recommended Scheme": "Nippon India Liquid Fund", "Status": "Target Allocation"},
            {"Bucket": "Intermediate Yield", "Allocated (₹)": f"₹{arb_alloc:,.2f}", "Recommended Scheme": "ICICI Pru Arbitrage Fund", "Status": "Target Allocation"},
            {"Bucket": "Fixed Income Anchor", "Allocated (₹)": f"₹{debt_alloc:,.2f}", "Recommended Scheme": "HDFC Short-Term Debt Fund", "Status": "Target Allocation"},
            {"Bucket": "Tactical Satellite", "Allocated (₹)": f"₹{tactical_satellite_alloc:,.2f}", "Recommended Scheme": "Momentum Stocks (Tab 2)", "Status": "Standby — Pending Signal"}
        ])
        st.dataframe(parking_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🎯 Systematic Core Equity Phase-in Schedule (Starting Sep 2026)")
    sip_schedule_df = pd.DataFrame([
        {
            "Market Cap Segment": "Large Cap / Index",
            "Target Weight": f"{large_cap_pct}%",
            "Weekly SIP (₹)": f"₹{sip_large:,.2f}",
            "Monthly Run-Rate (₹)": f"₹{sip_large * 4:,.2f}",
            "Recommended Direct-Growth Schemes": "UTI Nifty 50 Index Fund / HDFC Top 100 Fund",
            "Role in Portfolio": "Low Beta Core Anchor"
        },
        {
            "Market Cap Segment": "Flexi Cap",
            "Target Weight": f"{flexi_cap_pct}%",
            "Weekly SIP (₹)": f"₹{sip_flexi:,.2f}",
            "Monthly Run-Rate (₹)": f"₹{sip_flexi * 4:,.2f}",
            "Recommended Direct-Growth Schemes": "Parag Parikh Flexi Cap Fund",
            "Role in Portfolio": "All-Weather Blend"
        },
        {
            "Market Cap Segment": "Mid / Small Cap",
            "Target Weight": f"{mid_small_pct}%",
            "Weekly SIP (₹)": f"₹{sip_mid_small:,.2f}",
            "Monthly Run-Rate (₹)": f"₹{sip_mid_small * 4:,.2f}",
            "Recommended Direct-Growth Schemes": "HDFC Mid-Cap Opportunities / SBI Small Cap Fund",
            "Role in Portfolio": "High-Alpha Compounding"
        }
    ])
    st.dataframe(sip_schedule_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: TACTICAL SATELLITE & DYNAMIC SECTOR RADAR
# =============================================================================
with tab2:
    st.subheader("🎯 Tactical Stock Momentum & Dual-Gate Filter")
    universe_tickers, universe_meta = cached_universe(universe_choice, st.session_state.universe_bust)

    st.caption(
        f"Active Satellite Budget: **₹{tactical_satellite_alloc:,.2f}** ({tactical_satellite_pct}% of capital). "
        f"Screening **{len(universe_tickers)} constituents from {universe_choice}**."
    )

    m1, m2, m3 = st.columns(3)
    if macro_data_ok:
        m1.metric("Nifty 50 LTP", f"₹{n_price:,.2f}")
        m2.metric("Nifty 200-Day EMA", f"₹{n_ema:,.2f}")
        m3.metric("Macro Gate Regime", "🟢 BULLISH" if is_bull else "🔴 CAUTION / BEARISH")
    else:
        m1.metric("Nifty 50 LTP", "—")
        m2.metric("Nifty 200-Day EMA", "—")
        m3.metric("Macro Gate Regime", "⚪ DATA UNAVAILABLE")

    st.markdown("---")

    with st.spinner(f"Evaluating {len(universe_tickers)} tickers & calculating Sector Relative Strength..."):
        weights_tuple = tuple(sorted(scorecard_weights.items()))
        liquidity_floor_inr = liquidity_floor_cr * 10_000_000.0
        scorecard_df, top_n_df, scan_meta = cached_scorecard_scan(
            tuple(universe_tickers), tactical_satellite_alloc, top_n_stocks, weights_tuple,
            liquidity_floor_inr, max_leverage_multiple, st.session_state.scorecard_bust
        )

    tech_meta = scan_meta.get("technical", {})
    fund_meta = scan_meta.get("fundamentals", {})
    scorecard_meta = scan_meta.get("scorecard", {})
    sector_caps = scorecard_meta.get("sector_caps", {})

    render_skip_report(tech_meta, label="tickers (technical data)")
    render_skip_report(fund_meta, label="tickers (fundamentals data)")

    # =============================================================================
    # SECTOR STRENGTH RADAR & RELATIVE PERFORMANCE VISUALS
    # =============================================================================
    st.markdown("#### 📡 Sector Relative Strength & Dynamic Allocation Caps")
    st.caption("Measures 3-month relative performance against Nifty 50 to automatically expand or restrict sector exposure.")

    if sector_caps:
        col_rad1, col_rad2 = st.columns([1, 1])

        # Convert caps to chart dataframe
        sec_chart_df = pd.DataFrame([
            {
                "Sector": s,
                "Dynamic Cap (%)": round(c * 100, 1),
                "Regime": "🟢 Leading (30% Max)" if c >= 0.30 else ("🟡 Neutral (20% Max)" if c >= 0.20 else "🔴 Lagging (10% Max)")
            }
            for s, c in sector_caps.items()
        ])

        with col_rad1:
            # Radar Visual
            categories = list(sector_caps.keys())
            values = [c * 100 for c in sector_caps.values()]
            # Close the radar loop
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.3)',
                line=dict(color='rgb(59, 130, 246)', width=2),
                name='Sector Cap %'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 35], ticksuffix="%")
                ),
                showlegend=False,
                height=320,
                margin=dict(t=20, b=20, l=40, r=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_rad2:
            # Bar Visual for Caps & Status
            fig_bar = px.bar(
                sec_chart_df,
                x="Sector",
                y="Dynamic Cap (%)",
                color="Regime",
                color_discrete_map={
                    "🟢 Leading (30% Max)": "#22c55e",
                    "🟡 Neutral (20% Max)": "#eab308",
                    "🔴 Lagging (10% Max)": "#ef4444"
                },
                text="Dynamic Cap (%)"
            )
            fig_bar.update_layout(height=320, margin=dict(t=20, b=20, l=10, r=10), yaxis_range=[0, 35])
            fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 🧮 Scorecard Pipeline Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scanned Universe", f"{scorecard_meta.get('total', 0)} stocks",
              help="The full technical universe — every ticker that made it through get_technical_signals, whether or not it went on to fundamentals.")
    c2.metric("⏭️ Not Sent to Fundamentals", f"{scorecard_meta.get('not_prescreened', 0)}",
              help="Non-bullish (or otherwise not pre-screened) tickers — skipped BEFORE the expensive fundamentals fetch to save network calls, not silently dropped from this count.")
    c3.metric("🚫 Vetoed (Illiquidity)", f"{scorecard_meta.get('vetoed_illiquid', 0)}")
    c4.metric("🚫 Vetoed (Leverage)", f"{scorecard_meta.get('vetoed_leverage', 0)}")
    c5.metric("✅ Fully Scored & Ranked", f"{scorecard_meta.get('scored', 0)}")

    if not scorecard_df.empty:
        with st.expander("Full scorecard — every scanned stock, scored, vetoed, or pre-filtered out"):
            score_display_cols = [
                "Stock", "Composite Score", "Momentum Score", "Quality Score", "Valuation Score",
                "Liquidity Score", "Data Completeness", "Vetoed", "Veto Reason", "LTP (₹)", "Sector"
            ]
            valid_score_cols = [c for c in score_display_cols if c in scorecard_df.columns]
            st.dataframe(scorecard_df[valid_score_cols], use_container_width=True, hide_index=True)
            st.caption(
                "Rows marked \"Not Scored — outside momentum pre-filter\" were never sent to the "
                "fundamentals fetch because they weren't SuperTrend-bullish — they're kept here "
                "instead of silently dropped, so this table always reflects the FULL scanned universe."
            )

    st.markdown(f"#### 🎯 Top {top_n_stocks} Ranked Momentum Allocations (Sector Constrained)")
    if not top_n_df.empty:
        display_cols = ["Stock", "Sector", "Composite Score", "LTP (₹)", "Dynamic Stop (₹)", "Risk to Stop (%)",
                        "Weight (%)", "Allocation (₹)", "Qty to Buy", "Capital Actually Deployed (₹)",
                        "Data Completeness"]
        valid_cols = [c for c in display_cols if c in top_n_df.columns]
        st.dataframe(top_n_df[valid_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No stocks currently qualify under the strict momentum & fundamental filters.")

    st.markdown("---")

    # =============================================================================
    # ON-DEMAND SCREENER — single ticker / ETF / AMFI scheme lookup
    # =============================================================================
    st.subheader("🔎 On-Demand Screener")
    st.caption("Look up any single NSE ticker, ETF, or AMFI mutual fund scheme code for an instant verdict.")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        searched_ticker = st.text_input(
            "Enter Ticker, ETF, or AMFI Scheme Code (e.g. INFY, NIFTYBEES, 119062):",
            key="ondemand_ticker_input"
        )
    with col_s2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 Analyze Asset", use_container_width=True)

    # Store the result together with the exact query it was computed for --
    # otherwise editing the text box without re-clicking would silently keep
    # showing a PREVIOUS query's result next to the NEW text typed in.
    if analyze_btn and searched_ticker:
        with st.spinner(f"Analyzing {searched_ticker.upper()}..."):
            res = cached_single_asset(searched_ticker.strip(), st.session_state.refresh_counter)
        st.session_state["last_analyzed_query"] = searched_ticker.strip()
        st.session_state["last_analyzed_result"] = res

    stored_query = st.session_state.get("last_analyzed_query")
    res = st.session_state.get("last_analyzed_result")

    if res is not None and stored_query:
        st.caption(f"Showing result for: **{stored_query}**")
        if searched_ticker.strip() and searched_ticker.strip() != stored_query:
            st.info(f"ℹ️ The box currently shows a different value (**{searched_ticker.strip()}**) — click **Analyze Asset** to search for it.")

        if "error" in res:
            st.error(f"❌ {res['error']}")
        elif res.get("asset_type") == "mutual_fund":
            st.info("ℹ️ " + res.get("methodology_note", ""))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Latest NAV", f"₹{res['ltp']:,.2f}")
            k2.metric("200-Day EMA (NAV)", f"₹{res['ema_200']:,.2f}", delta="Above" if res["above_ema"] else "Below")
            k3.metric("3M Return", f"{res['ret_3m_pct']}%" if res["ret_3m_pct"] is not None else "—")
            k4.metric("6M Return", f"{res['ret_6m_pct']}%" if res["ret_6m_pct"] is not None else "—")

            if res["verdict"] == "GREEN":
                st.success(f"### {res['verdict_text']}")
            elif res["verdict"] == "YELLOW":
                st.warning(f"### {res['verdict_text']}")
            else:
                st.error(f"### {res['verdict_text']}")

            st.markdown("##### 📌 Recommendation")
            rec_colors = {"GREEN": st.success, "YELLOW": st.warning, "RED": st.error}
            rec_fn = rec_colors.get(res["verdict"], st.info)
            rec_fn(f"**{res.get('recommendation', '—')}**")

            exit_labels = {
                "HOLD": "✅ Trend Status: NO ACTION NEEDED",
                "MONITOR": "🟡 Trend Status: WATCH CLOSELY",
                "REVIEW": "🔴 Trend Status: REVIEW THIS HOLDING",
            }
            exit_signal = res.get("exit_signal", "")
            status_label = exit_labels.get(exit_signal, f"Trend Status: {exit_signal}")
            if exit_signal == "REVIEW":
                st.error(f"**{status_label}**\n\n{res.get('exit_note', '')}")
            elif exit_signal == "MONITOR":
                st.warning(f"**{status_label}**\n\n{res.get('exit_note', '')}")
            else:
                st.success(f"**{status_label}**\n\n{res.get('exit_note', '')}")
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("LTP", f"₹{res['ltp']:,.2f}", help=f"As of {res.get('as_of', '—')}")
            k2.metric("SuperTrend", "🟢 Bullish" if res["st_bullish"] else "🔴 Bearish")
            k3.metric("ADX", f"{res['adx']:.1f}")
            k4.metric("200 EMA", f"₹{res['ema_200']:,.2f}", delta="Above" if res["above_ema"] else "Below")
            k5.metric("Dynamic Stop", f"₹{res['dynamic_stop']:,.2f}", f"{res['risk_to_stop']}%")

            if res["verdict"] == "GREEN":
                st.success(f"### {res['verdict_text']}")
            elif res["verdict"] == "YELLOW":
                st.warning(f"### {res['verdict_text']}")
            else:
                st.error(f"### {res['verdict_text']}")

            st.markdown("##### 📌 Recommendation")
            rec_colors = {"GREEN": st.success, "YELLOW": st.warning, "RED": st.error}
            rec_fn = rec_colors.get(res["verdict"], st.info)
            rec_fn(f"**{res.get('recommendation', '—')}**\n\n{res.get('recommendation_note', '')}")

            st.markdown("##### 🎯 Risk-Based Position Sizing")
            st.caption(
                "The dynamic stop tells you WHERE to exit. This tells you HOW MANY shares to buy "
                "so that hitting the stop only costs you a fixed, chosen amount of capital — "
                "instead of sizing the position off gut feel."
            )
            risk_col1, risk_col2 = st.columns([1, 2])
            with risk_col1:
                risk_amount = st.number_input(
                    "Rupees you're willing to lose on this trade if the stop is hit (₹)",
                    min_value=100.0,
                    max_value=float(tactical_satellite_alloc) if tactical_satellite_alloc > 0 else 100000.0,
                    value=min(2500.0, max(100.0, tactical_satellite_alloc * 0.1)) if tactical_satellite_alloc > 0 else 2500.0,
                    step=250.0,
                    key="risk_amount_input"
                )
            with risk_col2:
                per_share_risk = res["ltp"] - res["dynamic_stop"]
                if per_share_risk > 0:
                    suggested_qty = int(risk_amount / per_share_risk)
                    suggested_capital = round(suggested_qty * res["ltp"], 2)
                    st.metric("Suggested Quantity", f"{suggested_qty} shares", help=f"Risk per share: ₹{per_share_risk:,.2f}")
                    st.caption(f"≈ ₹{suggested_capital:,.2f} deployed — if the dynamic stop is hit, realized loss ≈ ₹{round(suggested_qty * per_share_risk, 2):,.2f} (before brokerage/taxes).")
                else:
                    st.info("Stop is at or above current LTP — sizing math doesn't apply here (verify the SuperTrend/stop values before considering entry).")

    st.markdown("---")

    # =============================================================================
    # LIVE IPO CALENDAR (TAB 2) — real NSE data, nothing hardcoded
    # =============================================================================
    st.markdown("---")
    st.subheader("🚀 Live IPO Calendar")

    with st.spinner("Fetching live IPO calendar from NSE..."):
        ipo_df, ipo_raw, ipo_meta = cached_ipo_scan(st.session_state.refresh_counter)

    if not ipo_meta.get("ok"):
        st.error(
            f"⚠️ **Could not fetch live IPO data from NSE right now** "
            f"(`{ipo_meta.get('error', 'unknown error')}`). Nothing is shown below — "
            f"this tool never falls back to stale or hand-typed data. Try **🔄 Force Full Data Refresh** "
            f"in the sidebar in a minute."
        )
    elif ipo_df.empty:
        st.info(f"NSE reports no active/upcoming IPOs right now. (Checked {ipo_meta.get('as_of', '—')}.)")
    else:
        st.caption(
            f"🟢 Live as of **{ipo_meta.get('as_of', '—')}**, fetched directly from NSE (cached ~15 min). "
            f"Dates and status are computed against today's actual date — nothing here is hand-typed."
        )
        st.info(
            "💡 **What's shown vs. what isn't:** Issue dates, price band, lot size, issue size, and "
            "Fresh Issue % are sourced live from NSE. Subscription — Overall, QIB, NII/HNI, RII — is "
            "fetched live for OPEN issues when NSE's bid-detail endpoint responds. **Grey Market "
            "Premium and peer-P/E valuation are intentionally NOT shown** — no reliable free live "
            "source exists for them, and a guessed number would be worse than none. The Recommendation "
            "column is a rule built only from the live fields above (QIB subscription is weighted "
            "highest — institutions do real diligence, unlike retail) — it's a rule to weigh, not a "
            "guarantee, and it's transparent about exactly which live inputs it had for each issue."
        )

        reco_df = build_ipo_recommendation(ipo_df)

        open_count = int((reco_df["Status"] == "OPEN").sum())
        upcoming_count = int((reco_df["Status"] == "UPCOMING").sum())
        closed_count = int((reco_df["Status"] == "CLOSED").sum())

        i1, i2, i3 = st.columns(3)
        i1.metric("🟢 Open Now", f"{open_count} Issues")
        i2.metric("🟡 Upcoming", f"{upcoming_count} Issues")
        i3.metric("⚪ Closed", f"{closed_count} Issues")

        disp_cols = [
            "Company", "Symbol", "Segment", "Status", "Issue Opens", "Issue Closes", "Days to Close",
            "Price Band", "Lot Size", "Issue Size", "Fresh Issue (%)",
            "Sub — Overall (x)", "Sub — QIB (x)", "Sub — NII/HNI (x)", "Sub — RII (x)",
            "Recommendation", "Live Data Completeness"
        ]
        valid_disp_cols = [c for c in disp_cols if c in reco_df.columns]
        st.dataframe(reco_df[valid_disp_cols], use_container_width=True, hide_index=True)

        open_with_reco = reco_df[reco_df["Status"] == "OPEN"]
        if not open_with_reco.empty:
            with st.expander("📋 Recommendation reasoning — one line per open issue"):
                for _, row in open_with_reco.iterrows():
                    st.markdown(f"**{row['Company']}** — {row['Recommendation']}  \n{row['Recommendation Note']}")

        render_skip_report(ipo_meta, label="IPO listings")

        # If NSE's response came back non-empty but nothing actually parsed
        # (schema drift on their unofficial API), surface the raw payload
        # instead of silently showing an empty/wrong table.
        if ipo_raw and reco_df.empty:
            with st.expander("🔍 Diagnostic: raw NSE response (parser found no usable rows)"):
                st.json(ipo_raw[:3])
                st.caption(
                    "NSE returned data but none of it matched the field names this parser expects — "
                    "their API is unofficial and can change without notice. Share this raw shape to "
                    "get the field-mapping corrected."
                )

        # Subscription (QIB/NII/RII) and Lot Size / Fresh Issue % all come
        # from ONE endpoint now (ipo-detail). Surface its raw response if
        # any of those fields are still coming up universally blank, so a
        # wrong field-name guess is visible instead of another silent gap.
        detail_debug = ipo_meta.get("detail_debug")
        any_open = (reco_df["Status"] == "OPEN").any() if not reco_df.empty else False
        qib_missing = reco_df["Sub — QIB (x)"].isna().all() if "Sub — QIB (x)" in reco_df.columns else True
        nii_missing = reco_df["Sub — NII/HNI (x)"].isna().all() if "Sub — NII/HNI (x)" in reco_df.columns else True
        lot_size_unavailable = (
            reco_df["Lot Size"].astype(str).str.contains("Not published", na=False).all()
            if "Lot Size" in reco_df.columns else True
        )
        if detail_debug and any_open and (qib_missing or nii_missing or lot_size_unavailable):
            with st.expander("🔍 Diagnostic: live IPO detail fetch (subscription / Lot Size / Fresh Issue % gaps)"):
                st.write(f"**Symbol tested:** {detail_debug.get('symbol')}")
                st.write(f"**URL called:** {detail_debug.get('url')}")
                st.write(f"**HTTP status:** {detail_debug.get('status_code')}")
                st.write(
                    f"**QIB missing:** {qib_missing} · **NII/HNI missing:** {nii_missing} · "
                    f"**Lot Size unavailable:** {lot_size_unavailable}"
                )
                if detail_debug.get("exception"):
                    st.error(f"Request failed: {detail_debug['exception']}")
                elif detail_debug.get("raw_json") is not None:
                    st.json(detail_debug["raw_json"])
                elif detail_debug.get("raw_text_snippet"):
                    st.code(detail_debug["raw_text_snippet"], language="text")
                else:
                    st.warning(
                        "This endpoint returned nothing usable. It's possible NSE simply doesn't publish "
                        "some of this data anywhere in a free API — at that point it's not worth chasing "
                        "further, and the app will keep showing \"Not published by NSE's public feed\" "
                        "honestly rather than guessing."
                    )
                st.caption(
                    "This is NSE's actual response for the endpoint this tool uses. Share this output to "
                    "get the field-mapping corrected, or to confirm the gap is a real data limitation."
                )

# =============================================================================
# TAB 3: CORE MUTUAL FUND QUALIFIER
# =============================================================================
with tab3:
    st.subheader("📈 Core Mutual Fund Qualifier Engine")
    st.caption("Screens AMFI NAV series using 200 EMA Trend + Regression Jensen's Alpha (α) + Sortino Downside Protection.")

    col_btn, col_filter = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Refresh Fund Analysis", use_container_width=True):
            st.session_state.refresh_counter += 1
            st.cache_data.clear()
            st.rerun()
    with col_filter:
        show_qualified_only = st.checkbox("🎯 Show ONLY 'Clear to Accumulate' Qualifiers", value=True)

    with st.spinner("Executing live AMFI resolution and linear regression calculations..."):
        all_mf_df, mf_meta = cached_mf_scan(st.session_state.refresh_counter)

    render_skip_report(mf_meta, label="funds")

    if not all_mf_df.empty and "Verdict" in all_mf_df.columns:
        green_funds_by_cat = all_mf_df[all_mf_df["Verdict"] == "GREEN"].groupby("Category")["Scheme Name"].count().to_dict()

        def calc_fund_alloc(row):
            if row["Verdict"] != "GREEN": return 0.0
            cat = row["Category"]
            cat_b = 0.0
            if "Large Cap" in cat: cat_b = weekly_sip_amount * (large_cap_pct / 100.0)
            elif "Flexi Cap" in cat: cat_b = weekly_sip_amount * (flexi_cap_pct / 100.0)
            elif "Mid Cap" in cat: cat_b = (weekly_sip_amount * (mid_small_pct / 100.0)) / 2.0
            elif "Small Cap" in cat: cat_b = (weekly_sip_amount * (mid_small_pct / 100.0)) / 2.0
            num_q = green_funds_by_cat.get(cat, 1)
            return round(cat_b / max(1, num_q), 2)

        all_mf_df["Weekly SIP (₹)"] = all_mf_df.apply(calc_fund_alloc, axis=1)
        all_mf_df["Monthly SIP (₹)"] = all_mf_df["Weekly SIP (₹)"] * 4.0

        green_funds = len(all_mf_df[all_mf_df["Verdict"] == "GREEN"])
        total_funds = len(all_mf_df)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Scanned Universe", f"{total_funds} Schemes")
        col_f2.metric("🟢 Tri-Factor Qualified", f"{green_funds}")
        col_f3.metric("SIP Allocation Rate", f"₹{weekly_sip_amount:,.2f}/wk")

        display_df = all_mf_df[all_mf_df["Verdict"] == "GREEN"].reset_index(drop=True) if show_qualified_only else all_mf_df
        display_cols = ["Scheme Name", "Category", "Latest NAV (₹)", "200 EMA (₹)", "History (Yrs)", "CAGR (%)", "Alpha (α %)", "Beta (β)", "Sortino", "Weekly SIP (₹)", "Actionable Recommendation"]
        valid_cols = [c for c in display_cols if c in display_df.columns]
        st.dataframe(display_df[valid_cols], use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: SCENARIO SANDBOX & INTERACTIVE BACKTESTER
# =============================================================================
with tab4:
    st.subheader("🧪 Scenario Sandbox & Strategy Verification")
    
    tab4_sub1, tab4_sub2 = st.tabs(["📈 Multi-Year Growth Simulator", "🔬 Historical Momentum Backtester"])
    
    # SUB-TAB 1: GROWTH SIMULATOR
    with tab4_sub1:
        st.markdown("#### 🏛️ Wealth Accumulation Simulator")
        sim_years = st.slider("Investment Horizon (Years)", 1, 10, 5)
        col_s1, col_s2 = st.columns(2)
        exp_eq_cagr = col_s1.slider("Expected Equity CAGR (%)", 6.0, 20.0, 13.0, 0.5)
        exp_yd_cagr = col_s2.slider("Preservation Yield CAGR (%)", 4.0, 9.0, 7.0, 0.2)

        months_total = sim_years * 12
        monthly_sip = weekly_sip_amount * (52 / 12)
        m_eq_rate = (1 + exp_eq_cagr / 100.0) ** (1 / 12) - 1
        m_yd_rate = (1 + exp_yd_cagr / 100.0) ** (1 / 12) - 1

        sim_sat_init = total_managed_capital * (tactical_satellite_pct / 100.0)
        sim_feeder_init = max(0.0, total_managed_capital - emergency_liquid_alloc - sim_sat_init)

        months_axis, runway_series, feeder_series, equity_series, satellite_series, total_wealth_series = [], [], [], [], [], []
        curr_runway, curr_feeder, curr_equity, curr_satellite = emergency_liquid_alloc, sim_feeder_init, 0.0, sim_sat_init

        for m in range(months_total + 1):
            months_axis.append(m)
            runway_series.append(curr_runway)
            feeder_series.append(curr_feeder)
            equity_series.append(curr_equity)
            satellite_series.append(curr_satellite)
            total_wealth_series.append(curr_runway + curr_feeder + curr_equity + curr_satellite)

            if m < months_total:
                curr_runway = curr_runway * (1 + (m_yd_rate * 0.9))
                feeder_draw = min(curr_feeder, monthly_sip)
                curr_feeder = max(0.0, (curr_feeder - feeder_draw)) * (1 + m_yd_rate)
                curr_equity = (curr_equity + feeder_draw) * (1 + m_eq_rate)
                curr_satellite = curr_satellite * (1 + m_eq_rate * 1.1)

        sim_df = pd.DataFrame({
            "Year": [round(m / 12, 1) for m in months_axis],
            "Emergency Runway (₹)": runway_series,
            "Arbitrage/Debt Feeder (₹)": feeder_series,
            "Core Compounding Equity (₹)": equity_series,
            "Tactical Satellite (₹)": satellite_series,
            "Total Net Worth (₹)": total_wealth_series
        })

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=sim_df["Year"], y=sim_df["Emergency Runway (₹)"], name="Emergency Buffer", stackgroup="one", line=dict(width=0.5)))
        fig_sim.add_trace(go.Scatter(x=sim_df["Year"], y=sim_df["Arbitrage/Debt Feeder (₹)"], name="Preservation Yield Pool", stackgroup="one", line=dict(width=0.5)))
        fig_sim.add_trace(go.Scatter(x=sim_df["Year"], y=sim_df["Core Compounding Equity (₹)"], name="Core Equity MFs", stackgroup="one", line=dict(width=0.5)))
        fig_sim.add_trace(go.Scatter(x=sim_df["Year"], y=sim_df["Tactical Satellite (₹)"], name="Tactical Satellite", stackgroup="one", line=dict(width=0.5)))
        fig_sim.add_trace(go.Scatter(x=sim_df["Year"], y=sim_df["Total Net Worth (₹)"], name="Total Portfolio", line=dict(color="white", width=3, dash="dash")))

        fig_sim.update_layout(height=350, margin=dict(t=20, b=20, l=10, r=10), xaxis_title="Horizon (Years)", yaxis_title="Portfolio (₹)")
        st.plotly_chart(fig_sim, use_container_width=True)

    # SUB-TAB 2: HISTORICAL BACKTEST HARNESS CARD
    with tab4_sub2:
        st.markdown("#### 🔬 Empirical Strategy Backtesting")
        st.caption("Stress-tests the Dual-Gate momentum rules (SuperTrend + ADX + ATR Trailing Stops) across multi-year historical bars to calculate trade expectancy.")

        col_bt1, col_bt2, col_bt3 = st.columns(3)
        bt_adx = col_bt1.slider("ADX Momentum Filter", min_value=20.0, max_value=35.0, value=25.0, step=1.0)
        bt_hold = col_bt2.selectbox("Holding Horizon / Rebalance Period", [4, 8, 12, 16], index=1, format_func=lambda x: f"{x} Weeks (~{x//4} Months)")
        bt_lookback = col_bt3.slider("Lookback History (Years)", min_value=1, max_value=5, value=3, step=1)

        if st.button("🚀 Run Empirical Backtest Simulation", use_container_width=True):
            with st.spinner("Downloading historical price arrays and simulating trailing stops..."):
                universe_sample = universe_tickers[:25] if len(universe_tickers) > 25 else universe_tickers
                bt_results = run_historical_backtest(
                    universe_sample,
                    adx_threshold=bt_adx,
                    hold_period_weeks=bt_hold,
                    lookback_years=bt_lookback
                )

            if "error" in bt_results:
                st.error(f"❌ {bt_results['error']}")
            elif bt_results["total_trades"] == 0:
                st.warning("⚠️ Zero historical trade triggers found for this parameter combination.")
            else:
                st.success("### ✅ Historical Simulation Completed")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Simulated Trades", f"{bt_results['total_trades']}")
                b2.metric("Strategy Win Rate", f"{bt_results['win_rate']}")
                b3.metric("Avg Return / Trade", f"{bt_results['avg_return_pct']}")
                b4.metric("Profit Factor", f"{bt_results['profit_factor']}")

                trades_df = bt_results["trades_df"]
                st.markdown("##### 📋 Historical Trade Log (Sample)")
                st.dataframe(trades_df.head(20), use_container_width=True, hide_index=True)