import io
import re
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# 1. MACRO REGIME ENGINE (NIFTY 50 vs 200 EMA)
# =============================================================================
def get_nifty_macro_regime():
    """
    Fetches live Nifty 50 price and its 200-day EMA.
    Returns explicit error states on network failure to avoid false signals.
    """
    try:
        nifty = yf.download("^NSEI", period="2y", interval="1d", progress=False)
        if nifty.empty or len(nifty) < 200:
            return {"ok": False, "error": "Insufficient Nifty data (<200 sessions)."}

        close = nifty["Close"].squeeze()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        current_price = float(close.iloc[-1])
        current_ema = float(ema_200.iloc[-1])
        diff_pts = round(current_price - current_ema, 2)
        diff_pct = round((diff_pts / current_ema) * 100, 2) if current_ema else 0.0
        as_of = nifty.index[-1].strftime("%d-%b-%Y")

        return {
            "ok": True,
            "bullish": current_price > current_ema,
            "nifty_price": round(current_price, 2),
            "ema_200": round(current_ema, 2),
            "diff_pts": diff_pts,
            "diff_pct": diff_pct,
            "as_of": as_of,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _get_nifty_daily_returns():
    """Helper for regression beta calculation against Nifty 50."""
    try:
        nifty = yf.download("^NSEI", period="3y", interval="1d", progress=False)
        if nifty.empty:
            return None
        close = nifty["Close"].squeeze()
        rets = close.pct_change().dropna()
        rets.index = [d.strftime("%d-%m-%Y") for d in rets.index]
        return rets
    except Exception:
        return None


# =============================================================================
# 2. DYNAMIC SCAN UNIVERSE (Live NSE Constituent Ingestion)
# =============================================================================
NSE_INDEX_CSV_URLS = {
    "NIFTY 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
}

FALLBACK_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "SUNPHARMA.NS", "BAJFINANCE.NS",
    "HAL.NS", "BEL.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TRENT.NS", "TITAN.NS", "JSWSTEEL.NS", "SHRIRAMFIN.NS", "APOLLOHOSP.NS", "NIFTYBEES.NS",
]


def get_universe_tickers(index_name="NIFTY 100"):
    """Fetches constituent lists directly from NSE's official archives."""
    url = NSE_INDEX_CSV_URLS.get(index_name)
    if not url:
        return list(FALLBACK_WATCHLIST), {"source": "fallback", "reason": "Unknown index", "index_name": index_name}

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return list(FALLBACK_WATCHLIST), {"source": "fallback", "reason": f"HTTP {resp.status_code}", "index_name": index_name}

        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = next((c for c in ["Symbol", "SYMBOL", "symbol"] if c in df.columns), None)
        if symbol_col is None or df.empty:
            return list(FALLBACK_WATCHLIST), {"source": "fallback", "reason": "No Symbol column", "index_name": index_name}

        tickers = [f"{str(sym).strip().upper()}.NS" for sym in df[symbol_col].dropna().tolist() if str(sym).strip()]
        return tickers, {"source": "live", "index_name": index_name, "count": len(tickers)}
    except Exception as e:
        return list(FALLBACK_WATCHLIST), {"source": "fallback", "reason": str(e), "index_name": index_name}


# =============================================================================
# 3. TECHNICAL MOMENTUM & DUAL-GATE INDICATORS
# =============================================================================
def compute_supertrend(df, period=10, multiplier=3.0):
    high, low, close = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = pd.Series(tr).rolling(window=period).mean().values
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper, final_lower = np.full(n, np.nan), np.full(n, np.nan)
    trend = np.zeros(n)

    start = period
    if start >= n:
        return trend, np.full(n, np.nan), atr

    final_upper[start], final_lower[start] = basic_upper[start], basic_lower[start]
    trend[start] = 1 if close[start] > final_upper[start] else -1

    for i in range(start + 1, n):
        final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
        if trend[i-1] == 1:
            trend[i] = -1 if close[i] < final_lower[i] else 1
        else:
            trend[i] = 1 if close[i] > final_upper[i] else -1

    st_line = np.where(trend == 1, final_lower, final_upper)
    return trend, st_line, atr


def compute_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    denom = (plus_di + minus_di).abs()
    dx = ((plus_di - minus_di).abs() / denom.replace(0, np.nan)) * 100
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None


def get_technical_signals(tickers):
    """Batched download for weekly and daily historical candles."""
    scored, skipped = [], []
    if not tickers:
        return pd.DataFrame(), {"attempted": 0, "succeeded": 0, "skipped": []}

    try:
        w_data = yf.download(tickers=tickers, period="2y", interval="1wk", group_by="ticker", progress=False, threads=True)
        d_data = yf.download(tickers=tickers, period="1y", interval="1d", group_by="ticker", progress=False, threads=True)
    except Exception as e:
        return pd.DataFrame(), {"attempted": len(tickers), "succeeded": 0, "skipped": [(t, str(e)) for t in tickers]}

    single = len(tickers) == 1
    for t in tickers:
        try:
            w = w_data if single else (w_data[t] if t in w_data.columns.get_level_values(0) else pd.DataFrame())
            d = d_data if single else (d_data[t] if t in d_data.columns.get_level_values(0) else pd.DataFrame())

            if w.empty or len(w) < 30 or d.empty:
                skipped.append((t, "Insufficient history"))
                continue

            clean_df = pd.DataFrame({'High': w['High'].squeeze(), 'Low': w['Low'].squeeze(), 'Close': w['Close'].squeeze()}).dropna()
            if len(clean_df) < 30:
                continue

            trend, st_line, _ = compute_supertrend(clean_df)
            adx = compute_adx(clean_df)
            ltp = float(clean_df['Close'].iloc[-1])
            stop = float(st_line[-1])
            is_bullish = bool(trend[-1] == 1)

            d_close = d["Close"].squeeze().dropna()
            ema200 = float(d_close.ewm(span=200, adjust=False).mean().iloc[-1])
            above_ema = bool(ltp > ema200)

            has_valid_stop = bool(is_bullish and ltp > 0 and 0 < stop < ltp)
            risk_pct = round(((ltp - stop) / ltp) * 100, 2) if has_valid_stop else None

            scored.append({
                "Stock": t.replace(".NS", ""),
                "Ticker": t,
                "LTP (₹)": round(ltp, 2),
                "SuperTrend Bullish": is_bullish,
                "ADX": round(adx, 1) if adx else 0.0,
                "Above 200 EMA": above_ema,
                "Dynamic Stop (₹)": round(stop, 2) if has_valid_stop else None,
                "Risk to Stop (%)": risk_pct,
            })
        except Exception as e:
            skipped.append((t, str(e)))

    return pd.DataFrame(scored), {"attempted": len(tickers), "succeeded": len(scored), "skipped": skipped}


# =============================================================================
# 4. EXTENDED FUNDAMENTALS & SCORECARD PIPELINE
# =============================================================================
def _normalize_debt_to_equity(val):
    if val is None: return None
    try: v = float(val)
    except: return None
    return round(v / 100.0, 3) if v > 10 else round(v, 3)

def _normalize_div_yield(val):
    if val is None: return None
    try: v = float(val)
    except: return None
    return round(v * 100.0, 2) if v < 1 else round(v, 2)

def _compute_roic_and_coverage(ticker):
    try:
        tk = yf.Ticker(ticker)
        fin, bs = tk.financials, tk.balance_sheet
        if fin is None or bs is None or fin.empty or bs.empty:
            return None, None

        def _row(df, keys):
            for k in keys:
                if k in df.index: return df.loc[k]
            return None

        ebit_row = _row(fin, ["EBIT", "Operating Income"])
        int_row = _row(fin, ["Interest Expense", "InterestExpense"])
        tax_row = _row(fin, ["Tax Provision", "Income Tax Expense"])
        pretax_row = _row(fin, ["Pretax Income", "Income Before Tax"])
        debt_row = _row(bs, ["Total Debt", "Short Long Term Debt Total"])
        eq_row = _row(bs, ["Stockholders Equity", "Common Stock Equity"])
        cash_row = _row(bs, ["Cash And Cash Equivalents", "Cash"])

        if ebit_row is None or pd.isna(ebit_row.iloc[0]): return None, None
        ebit = float(ebit_row.iloc[0])

        int_cov = None
        if int_row is not None and pd.notna(int_row.iloc[0]):
            iexp = abs(float(int_row.iloc[0]))
            if iexp > 0: int_cov = round(ebit / iexp, 2)

        roic = None
        if debt_row is not None and eq_row is not None and pretax_row is not None and tax_row is not None:
            t_debt = float(debt_row.iloc[0]) if pd.notna(debt_row.iloc[0]) else 0.0
            t_eq = float(eq_row.iloc[0]) if pd.notna(eq_row.iloc[0]) else None
            t_cash = float(cash_row.iloc[0]) if (cash_row is not None and pd.notna(cash_row.iloc[0])) else 0.0
            pretax = float(pretax_row.iloc[0]) if pd.notna(pretax_row.iloc[0]) else None

            if t_eq and pretax:
                inv_cap = t_debt + t_eq - t_cash
                tax_rate = float(tax_row.iloc[0]) / pretax if pd.notna(tax_row.iloc[0]) else None
                if tax_rate and 0.0 <= tax_rate <= 0.5 and inv_cap > 0:
                    nopat = ebit * (1 - tax_rate)
                    roic = round((nopat / inv_cap) * 100, 2)

        return roic, int_cov
    except:
        return None, None

def _fetch_single_fund(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info: return ticker, None, "No info"
        roic, int_cov = _compute_roic_and_coverage(ticker)
        return ticker, {
            "Stock": ticker.replace(".NS", ""),
            "Sector": info.get("sector", "Unknown"),
            "PE": round(info["trailingPE"], 2) if info.get("trailingPE") else None,
            "Debt/Equity": _normalize_debt_to_equity(info.get("debtToEquity")),
            "Dividend Yield (%)": _normalize_div_yield(info.get("dividendYield")),
            "P/B (CMP/BV)": round(info["priceToBook"], 2) if info.get("priceToBook") else None,
            "Avg Volume": float(info["averageVolume"]) if info.get("averageVolume") else None,
            "Earnings Growth 1Y (%)": round(info["earningsGrowth"] * 100, 2) if info.get("earningsGrowth") else None,
            "ROIC (%)": roic,
            "Interest Coverage (x)": int_cov,
        }, None
    except Exception as e:
        return ticker, None, str(e)

def get_extended_fundamentals(tickers, max_workers=8):
    if not tickers:
        return pd.DataFrame(), {"attempted": 0, "succeeded": 0, "skipped": []}
    rows, skipped = [], []
    
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yfin_fund")
    try:
        futures = {executor.submit(_fetch_single_fund, t): t for t in tickers}
        for f in as_completed(futures):
            ticker, row, err = f.result()
            if row:
                rows.append(row)
            else:
                skipped.append((ticker, err))
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False)
        
    return pd.DataFrame(rows), {"attempted": len(tickers), "succeeded": len(rows), "skipped": skipped}


DEFAULT_SCORECARD_WEIGHTS = {"momentum": 0.40, "quality": 0.30, "valuation": 0.20, "liquidity": 0.10}

def _percentile_rank(series, higher_is_better=True):
    valid = series.dropna()
    if len(valid) < 2: return series.apply(lambda x: 50.0 if pd.notna(x) else np.nan)
    ranks = series.rank(pct=True, na_option="keep") * 100.0
    return ranks if higher_is_better else (100.0 - ranks)

def compute_scorecard(tech_df, fund_df, weights=None, liquidity_floor_inr=10_000_000.0, max_leverage_multiple=3.0):
    """
    Merges technical + fundamentals data into one scorecard.

    tech_df may legitimately contain MORE tickers than fund_df: callers
    (see run_scorecard_scan) only send bullish candidates to the expensive
    extended-fundamentals fetch, to cut network calls. This uses a LEFT
    merge on tech_df so every technically-scanned ticker survives into the
    output -- non-screened tickers get an explicit "Not Scored — outside
    momentum pre-filter" status instead of silently vanishing the way an
    inner merge would. `total` in the returned meta is always len(tech_df)
    -- the TRUE scanned universe -- never just however many made it to
    fundamentals, so the UI can't understate how much was actually looked at.
    """
    weights = weights or DEFAULT_SCORECARD_WEIGHTS
    empty_meta = {"total": 0, "not_prescreened": 0, "vetoed_illiquid": 0, "vetoed_leverage": 0, "scored": 0}
    if tech_df.empty:
        return pd.DataFrame(), empty_meta

    fund_has_data = (not fund_df.empty) and ("Stock" in fund_df.columns)
    if fund_has_data:
        df = tech_df.merge(fund_df, on="Stock", how="left", indicator="_merge_src").reset_index(drop=True)
        has_fundamentals = df["_merge_src"] == "both"
        df = df.drop(columns=["_merge_src"])
    else:
        # Nothing came back from the fundamentals fetch at all (e.g. every
        # candidate errored) -- still return every technical row, just with
        # no ticker marked as fundamentals-screened.
        df = tech_df.copy().reset_index(drop=True)
        has_fundamentals = pd.Series(False, index=df.index)

    # Fundamentals columns may be entirely absent if fund_df was empty --
    # make sure downstream code always has them to reference.
    for c in ["Sector", "PE", "Debt/Equity", "Dividend Yield (%)", "P/B (CMP/BV)",
              "Avg Volume", "Earnings Growth 1Y (%)", "ROIC (%)", "Interest Coverage (x)"]:
        if c not in df.columns:
            df[c] = np.nan

    not_prescreened_mask = ~has_fundamentals

    def leave_one_out_median(col):
        valid = df[col].notna() & (df[col] > 0)
        res = []
        for idx, sec in zip(df.index, df["Sector"]):
            peers = df[valid & (df["Sector"] == sec) & (df.index != idx)]
            res.append(peers[col].median() if len(peers) >= 2 else np.nan)
        return pd.Series(res, index=df.index)

    df["PE Relative"] = df["PE"] / leave_one_out_median("PE")
    df["D/E Relative"] = df["Debt/Equity"] / leave_one_out_median("Debt/Equity")
    df["ADTV (₹)"] = df["Avg Volume"] * df["LTP (₹)"]

    illiquid_mask = df["ADTV (₹)"].notna() & (df["ADTV (₹)"] < liquidity_floor_inr)
    extreme_lev_mask = df["D/E Relative"].notna() & (df["D/E Relative"] > max_leverage_multiple)
    # Can't veto a stock on fundamentals data that was never fetched --
    # those rows are already excluded from scoring via not_prescreened_mask.
    vetoed = (illiquid_mask | extreme_lev_mask) & has_fundamentals

    df["Vetoed"] = vetoed
    df["Veto Reason"] = np.where(
        not_prescreened_mask, "Not Scored — outside momentum pre-filter (fundamentals not fetched)",
        np.where(illiquid_mask & extreme_lev_mask, "Illiquid + Leverage",
        np.where(illiquid_mask, "Illiquid (ADTV below floor)",
        np.where(extreme_lev_mask, "Extreme Leverage", "—")))
    )

    for c in ["Momentum Score", "Quality Score", "Valuation Score", "Liquidity Score", "Composite Score"]:
        df[c] = np.nan
    df["Data Completeness"] = "0/11"

    scoreable = df[~vetoed & ~not_prescreened_mask].copy()
    meta = {
        "total": len(df),
        "not_prescreened": int(not_prescreened_mask.sum()),
        "vetoed_illiquid": int(illiquid_mask.sum()),
        "vetoed_leverage": int(extreme_lev_mask.sum()),
        "scored": len(scoreable),
    }

    if scoreable.empty:
        return df.sort_values("Stock").reset_index(drop=True), meta

    scoreable["_adx_pct"] = _percentile_rank(scoreable["ADX"], True)
    scoreable["_roic_pct"] = _percentile_rank(scoreable["ROIC (%)"], True)
    scoreable["_intcov_pct"] = _percentile_rank(scoreable["Interest Coverage (x)"], True)
    scoreable["_de_rel_pct"] = _percentile_rank(scoreable["D/E Relative"], False)
    scoreable["_growth_pct"] = _percentile_rank(scoreable["Earnings Growth 1Y (%)"], True)
    scoreable["_pe_rel_pct"] = _percentile_rank(scoreable["PE Relative"], False)
    scoreable["_pb_pct"] = _percentile_rank(scoreable["P/B (CMP/BV)"], False)
    scoreable["_dy_pct"] = _percentile_rank(scoreable["Dividend Yield (%)"], True)
    scoreable["_adtv_pct"] = _percentile_rank(scoreable["ADTV (₹)"], True)

    def b_score(comps):
        avail = [(v, w) for v, w in comps if pd.notna(v)]
        if not avail: return np.nan, 0
        return sum(v * w for v, w in avail) / sum(w for _, w in avail), len(avail)

    mom_l, qual_l, val_l, liq_l, comp_l, comp_n = [], [], [], [], [], []
    for _, r in scoreable.iterrows():
        m_s, m_n = b_score([(r["_adx_pct"], 0.5), (100.0 if r["SuperTrend Bullish"] else 0.0, 0.25), (100.0 if r["Above 200 EMA"] else 0.0, 0.25)])
        q_s, q_n = b_score([(r["_roic_pct"], 0.4), (r["_intcov_pct"], 0.3), (r["_de_rel_pct"], 0.2), (r["_growth_pct"], 0.1)])
        v_s, v_n = b_score([(r["_pe_rel_pct"], 0.4), (r["_pb_pct"], 0.4), (r["_dy_pct"], 0.2)])
        l_s, l_n = b_score([(r["_adtv_pct"], 1.0)])

        b_vals = [ (m_s, weights["momentum"]), (q_s, weights["quality"]), (v_s, weights["valuation"]), (l_s, weights["liquidity"]) ]
        avail_b = [(v, w) for v, w in b_vals if pd.notna(v)]
        composite = (sum(v * w for v, w in avail_b) / sum(w for _, w in avail_b)) if avail_b else np.nan

        mom_l.append(m_s); qual_l.append(q_s); val_l.append(v_s); liq_l.append(l_s); comp_l.append(composite); comp_n.append(m_n + q_n + v_n + l_n)

    scoreable["Momentum Score"] = [round(v, 1) if pd.notna(v) else np.nan for v in mom_l]
    scoreable["Quality Score"] = [round(v, 1) if pd.notna(v) else np.nan for v in qual_l]
    scoreable["Valuation Score"] = [round(v, 1) if pd.notna(v) else np.nan for v in val_l]
    scoreable["Liquidity Score"] = [round(v, 1) if pd.notna(v) else np.nan for v in liq_l]
    scoreable["Composite Score"] = [round(v, 1) if pd.notna(v) else np.nan for v in comp_l]
    scoreable["Data Completeness"] = [f"{n}/11" for n in comp_n]

    cols = ["Momentum Score", "Quality Score", "Valuation Score", "Liquidity Score", "Composite Score", "Data Completeness"]
    df.loc[scoreable.index, cols] = scoreable[cols]
    return df.sort_values("Composite Score", ascending=False, na_position="last").reset_index(drop=True), meta


# =============================================================================
# 5. DYNAMIC SECTOR RELATIVE STRENGTH & CONSTRAINED ALLOCATION
# =============================================================================
SECTOR_BENCHMARKS = {
    "Financial Services": "^NSEBANK",
    "Technology": "^CNXIT",
    "Automobile": "^CNXAUTO",
    "Healthcare": "^CNXPHARMA",
    "Basic Materials": "^CNXMETAL",
    "Energy": "^CNXENERGY",
    "Industrials": "^CNXINFRA",
    "Consumer Cyclical": "^CNXCONSUM"
}

def compute_dynamic_sector_caps():
    """
    Computes dynamic allocation caps (max % of tactical budget) per sector
    by measuring 3-month relative performance against Nifty 50.
    """
    caps = {}
    try:
        nifty = yf.download("^NSEI", period="6m", interval="1d", progress=False)["Close"].squeeze()
        if len(nifty) < 63:
            return {s: 0.20 for s in SECTOR_BENCHMARKS.keys()}
            
        nifty_ret = (nifty.iloc[-1] / nifty.iloc[-63]) - 1.0

        for sector_name, sym in SECTOR_BENCHMARKS.items():
            try:
                sec_data = yf.download(sym, period="6m", interval="1d", progress=False)["Close"].squeeze()
                if sec_data.empty or len(sec_data) < 63:
                    caps[sector_name] = 0.20
                    continue
                
                sec_ret = (sec_data.iloc[-1] / sec_data.iloc[-63]) - 1.0
                rel_strength = (sec_ret - nifty_ret) * 100.0

                if rel_strength >= 3.0:
                    caps[sector_name] = 0.30   # Outperforming sector: 30% cap
                elif rel_strength >= -2.0:
                    caps[sector_name] = 0.20   # Neutral sector: 20% cap
                else:
                    caps[sector_name] = 0.10   # Lagging sector: 10% cap
            except Exception:
                caps[sector_name] = 0.20
    except Exception:
        for s in SECTOR_BENCHMARKS.keys():
            caps[s] = 0.20

    return caps


def run_scorecard_scan(tickers, satellite_budget=27000.0, top_n=15, weights=None,
                        liquidity_floor_inr=10_000_000.0, max_leverage_multiple=3.0):
    """
    Scorecard pipeline with dynamic sector concentration capping and integer allocation.
    """
    tech_df, tech_meta = get_technical_signals(tickers)
    if tech_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"technical": tech_meta, "fundamentals": {}, "scorecard": {}, "allocation": {}}

    bullish_candidates = tech_df[tech_df["SuperTrend Bullish"]]["Ticker"].tolist()
    tickers_to_fund_screen = bullish_candidates if len(bullish_candidates) >= 5 else tickers[:25]
    fund_df, fund_meta = get_extended_fundamentals(tickers_to_fund_screen)

    scorecard_df, scorecard_meta = compute_scorecard(
        tech_df, fund_df, weights=weights,
        liquidity_floor_inr=liquidity_floor_inr, max_leverage_multiple=max_leverage_multiple,
    )

    sector_caps = compute_dynamic_sector_caps()
    scorecard_meta["sector_caps"] = sector_caps

    scored_candidates = scorecard_df[scorecard_df["Composite Score"].notna()].copy()

    # Apply sector concentration limits
    selected_rows = []
    sector_counts = {}
    max_per_sector = max(2, int(top_n * 0.30))

    for _, row in scored_candidates.iterrows():
        sec = row.get("Sector", "Unknown")
        count = sector_counts.get(sec, 0)
        
        allowed_count = max_per_sector if sector_caps.get(sec, 0.20) >= 0.20 else max(1, int(top_n * 0.10))
        if count < allowed_count:
            selected_rows.append(row)
            sector_counts[sec] = count + 1
        
        if len(selected_rows) >= top_n:
            break

    top_n_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    # Affordability-aware integer allocation loop
    if not top_n_df.empty:
        survivors = top_n_df.copy()
        excluded_unaffordable = []
        per_stock_budget = 0.0

        for _ in range(len(top_n_df) + 1):
            n_surv = len(survivors)
            if n_surv == 0:
                break
            per_stock_budget = satellite_budget / n_surv
            unaff_mask = survivors["LTP (₹)"] > per_stock_budget
            if not unaff_mask.any():
                break
            excluded_unaffordable.extend(survivors.loc[unaff_mask, "Stock"].tolist())
            survivors = survivors[~unaff_mask].copy()

        top_n_df["Excluded — Price Exceeds Equal-Split Budget"] = top_n_df["Stock"].isin(excluded_unaffordable)

        if len(survivors) > 0:
            eq_weight = round(100.0 / len(survivors), 1)
            qty = (per_stock_budget / survivors["LTP (₹)"]).astype(int)
            deployed = qty * survivors["LTP (₹)"]

            top_n_df["Weight (%)"] = top_n_df["Stock"].map(dict(zip(survivors["Stock"], [f"{eq_weight}%"] * len(survivors)))).fillna("0% (excluded)")
            top_n_df["Allocation (₹)"] = top_n_df["Stock"].map(dict(zip(survivors["Stock"], [round(per_stock_budget, 2)] * len(survivors)))).fillna(0.0)
            top_n_df["Qty to Buy"] = top_n_df["Stock"].map(dict(zip(survivors["Stock"], qty))).fillna(0).astype(int)
            top_n_df["Capital Actually Deployed (₹)"] = top_n_df["Stock"].map(dict(zip(survivors["Stock"], deployed.round(2)))).fillna(0.0)
            total_deployed = round(deployed.sum(), 2)
        else:
            top_n_df["Weight (%)"], top_n_df["Allocation (₹)"], top_n_df["Qty to Buy"], top_n_df["Capital Actually Deployed (₹)"] = "0%", 0.0, 0, 0.0
            total_deployed = 0.0

        allocation_meta = {
            "satellite_budget": satellite_budget,
            "n_survivors": len(survivors),
            "excluded_unaffordable": excluded_unaffordable,
            "per_stock_budget": round(per_stock_budget, 2) if len(survivors) > 0 else 0.0,
            "total_deployed": total_deployed,
            "total_idle_cash": round(satellite_budget - total_deployed, 2),
        }
    else:
        allocation_meta = {"satellite_budget": satellite_budget, "n_survivors": 0, "excluded_unaffordable": [], "per_stock_budget": 0.0, "total_deployed": 0.0, "total_idle_cash": satellite_budget}

    meta = {"technical": tech_meta, "fundamentals": fund_meta, "scorecard": scorecard_meta, "allocation": allocation_meta}
    return scorecard_df, top_n_df, meta


def scan_strong_buys_with_allocation(tickers, satellite_budget=27000.0):
    """Dual-Gate fallback momentum scanner."""
    tech_df, meta = get_technical_signals(tickers)
    if tech_df.empty:
        return pd.DataFrame(), pd.DataFrame(), meta

    tech_df["Dynamic Stop (₹)"] = tech_df["Dynamic Stop (₹)"].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "—")
    tech_df["Risk to Stop (%)"] = tech_df["Risk to Stop (%)"].apply(lambda x: f"-{x}%" if pd.notna(x) else "—")
    tech_df["SuperTrend"] = tech_df["SuperTrend Bullish"].apply(lambda x: "🟢 Bullish" if x else "🔴 Bearish")
    tech_df["Above 200 EMA"] = tech_df["Above 200 EMA"].apply(lambda x: "✅ Yes" if x else "❌ No")
    tech_df["Qualified"] = (tech_df["SuperTrend"] == "🟢 Bullish") & (tech_df["ADX"] >= 25.0) & (tech_df["Above 200 EMA"] == "✅ Yes")

    top_10 = tech_df.sort_values(by="ADX", ascending=False).head(10).reset_index(drop=True)
    qualified = tech_df[tech_df["Qualified"]].copy().reset_index(drop=True)

    if not qualified.empty:
        n_q = len(qualified)
        eq_w = round(100.0 / n_q, 1)
        per_alloc = round(satellite_budget / n_q, 2)
        qualified["Weight (%)"] = f"{eq_w}%"
        qualified["Allocation (₹)"] = per_alloc
        qualified["Qty to Buy"] = (per_alloc / qualified["LTP (₹)"]).astype(int)
    return top_10, qualified, meta


# =============================================================================
# 6. CORE MUTUAL FUND RESOLVER & REGRESSION ENGINE
# =============================================================================
CORE_MF_UNIVERSE = [
    {"search_query": "UTI Nifty 50 Index Fund", "label": "UTI Nifty 50 Index Fund", "category": "Large Cap / Index", "bench_cagr": 12.0, "validate_keywords": ["uti", "nifty 50", "index"]},
    {"search_query": "HDFC Top 100 Fund", "label": "HDFC Top 100 Fund", "category": "Large Cap", "bench_cagr": 12.0, "validate_keywords": ["hdfc", "top 100"]},
    {"search_query": "Parag Parikh Flexi Cap Fund", "label": "Parag Parikh Flexi Cap Fund", "category": "Flexi Cap", "bench_cagr": 12.5, "validate_keywords": ["parag parikh", "flexi cap"]},
    {"search_query": "HDFC Mid-Cap Opportunities Fund", "label": "HDFC Mid-Cap Opportunities Fund", "category": "Mid Cap", "bench_cagr": 14.0, "validate_keywords": ["hdfc", "mid"]},
    {"search_query": "SBI Small Cap Fund", "label": "SBI Small Cap Fund", "category": "Small Cap", "bench_cagr": 14.5, "validate_keywords": ["sbi", "small cap"]},
    {"search_query": "ICICI Prudential Arbitrage Fund", "label": "ICICI Prudential Arbitrage Fund", "category": "Arbitrage / Feeder", "bench_cagr": 6.5, "validate_keywords": ["icici", "arbitrage"]},
    {"search_query": "HDFC Short Term Debt Fund", "label": "HDFC Short Term Debt Fund", "category": "Short Term Debt", "bench_cagr": 6.8, "validate_keywords": ["hdfc", "short term"]},
]

def resolve_scheme_code(search_query, validate_keywords=None, prefer_direct_growth=True, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get("https://api.mfapi.in/mf/search", params={"q": search_query}, timeout=12)
            if resp.status_code in {502, 503, 504, 429}:
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code != 200: return None, f"HTTP {resp.status_code}"
            matches = resp.json()
            if not matches: return None, "No matches"

            candidates = matches
            if validate_keywords:
                candidates = [m for m in candidates if all(kw.lower() in m.get("schemeName", "").lower() for kw in validate_keywords)] or candidates
            if prefer_direct_growth:
                candidates = [m for m in candidates if "direct" in m.get("schemeName", "").lower() and "growth" in m.get("schemeName", "").lower()] or candidates

            best = candidates[0]
            return {"code": str(best["schemeCode"]), "name": best["schemeName"]}, None
        except Exception:
            time.sleep(1.5 * (attempt + 1))
            continue
    return None, "Failed after retries"

def scan_core_mutual_funds():
    results, skipped = [], []
    risk_free_rate = 0.065
    nifty_rets = _get_nifty_daily_returns()

    for i, item in enumerate(CORE_MF_UNIVERSE):
        if i > 0: time.sleep(0.5)
        resolved, resolve_err = resolve_scheme_code(item["search_query"], item.get("validate_keywords"))
        if not resolved:
            skipped.append((item["label"], f"Unresolved code: {resolve_err}"))
            continue

        code = resolved["code"]
        try:
            resp = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=12)
            if resp.status_code != 200: continue
            payload = resp.json()
            actual_name = payload.get("meta", {}).get("scheme_name", "")
            data = payload.get("data", [])
            if len(data) < 250:
                skipped.append((item["label"], "Insufficient NAV points (<250)"))
                continue

            data_chrono = data[:750][::-1]
            s = pd.Series([float(d["nav"]) for d in data_chrono], index=[d["date"] for d in data_chrono])
            latest_nav = s.iloc[-1]
            ema_200 = s.ewm(span=200, adjust=False).mean().iloc[-1]

            ret_1y = round(((latest_nav / s.iloc[-250]) - 1) * 100, 1) if len(s) >= 250 else None
            span_yrs = round(len(s) / 250.0, 1)
            ret_cagr = round((((latest_nav / s.iloc[0]) ** (1 / span_yrs)) - 1) * 100, 2) if span_yrs > 0 else ret_1y

            d_rets = s.pct_change().dropna()
            neg_rets = d_rets[d_rets < 0]
            down_vol = (neg_rets.std() * np.sqrt(250)) if len(neg_rets) > 5 else None
            excess = (ret_cagr / 100.0) - risk_free_rate
            sortino = round(excess / down_vol, 2) if down_vol and down_vol > 0 else None

            is_cash = ("Arbitrage" in item["category"] or "Debt" in item["category"])
            beta, alpha_val = None, None

            if not is_cash and nifty_rets is not None:
                joined = pd.concat([d_rets.rename("fund"), nifty_rets.rename("bench")], axis=1, join="inner").dropna()
                if len(joined) >= 60:
                    cov, var = joined["fund"].cov(joined["bench"]), joined["bench"].var()
                    if var > 0:
                        beta = round(cov / var, 2)
                        capm_exp = risk_free_rate + beta * ((item["bench_cagr"] / 100.0) - risk_free_rate)
                        alpha_val = round(((ret_cagr / 100.0) - capm_exp) * 100, 2)

            above_ema = latest_nav > ema_200
            if is_cash: verdict, action = "GREEN", "🔒 Capital Preservation Anchor (Yield Pool)"
            elif beta is None or alpha_val is None or sortino is None: verdict, action = "GREY", "⚪ INCOMPLETE DATA"
            elif above_ema and sortino >= 1.2 and alpha_val >= 1.0: verdict, action = "GREEN", f"🟢 High Alpha & Protected (Sortino {sortino})"
            elif above_ema: verdict, action = "YELLOW", f"🟡 Compounding Unit (Moderate Alpha {alpha_val}%)"
            else: verdict, action = "RED", "🔴 Below 200 EMA (Pause Lumpsums)"

            results.append({
                "Scheme Name": actual_name, "AMFI Code": code, "Category": item["category"],
                "Latest NAV (₹)": round(latest_nav, 2), "200 EMA (₹)": round(ema_200, 2), "History (Yrs)": span_yrs,
                "1Y Ret (%)": f"{ret_1y}%" if ret_1y else "—", "CAGR (%)": f"{ret_cagr}%" if ret_cagr else "—",
                "Alpha (α %)": f"+{alpha_val}%" if alpha_val and alpha_val > 0 else f"{alpha_val}%" if alpha_val else "—",
                "Beta (β)": beta or "—", "Sortino": sortino or "—", "Verdict": verdict, "Actionable Recommendation": action
            })
        except Exception as e:
            skipped.append((item["label"], str(e)))

    return pd.DataFrame(results), {"attempted": len(CORE_MF_UNIVERSE), "succeeded": len(results), "skipped": skipped}


# =============================================================================
# 7. SINGLE ASSET DIAGNOSTIC ENGINE
# =============================================================================
def analyze_single_asset(ticker_or_code):
    ticker_str = str(ticker_or_code).strip().upper()
    if ticker_str.isdigit():
        try:
            r = requests.get(f"https://api.mfapi.in/mf/{ticker_str}", timeout=8)
            if r.status_code != 200: return {"error": f"HTTP {r.status_code}"}
            js = r.json()
            scheme_name = js.get("meta", {}).get("scheme_name", "")
            data = js.get("data", [])
            if len(data) < 200: return {"error": "Insufficient history (<200)"}

            data_chrono = data[:400][::-1]
            s = pd.Series([float(x["nav"]) for x in data_chrono])
            ltp = float(s.iloc[-1])
            ema_series = s.ewm(span=200, adjust=False).mean()
            ema_200 = float(ema_series.iloc[-1])
            above_ema = ltp > ema_200
            ret_3m = round(((ltp / s.iloc[-63]) - 1) * 100, 2) if len(s) >= 63 else None
            ret_6m = round(((ltp / s.iloc[-126]) - 1) * 100, 2) if len(s) >= 126 else None

            consecutive = 0
            curr_st = bool((s > ema_series).iloc[-1])
            for v in (s > ema_series).iloc[::-1]:
                if bool(v) == curr_st: consecutive += 1
                else: break

            trend_broken = (not curr_st) and (consecutive >= 15)
            exit_sig = "REVIEW" if trend_broken else ("MONITOR" if not curr_st else "HOLD")
            exit_note = f"NAV below 200 EMA for {consecutive} sessions." if not curr_st else f"NAV above 200 EMA for {consecutive} sessions."

            v_code = "GREEN" if (above_ema and ret_3m and ret_3m > 0) else ("YELLOW" if above_ema else "RED")
            return {
                "asset_type": "mutual_fund", "scheme_name": scheme_name, "fund_house": js.get("meta", {}).get("fund_house", ""),
                "ltp": round(ltp, 2), "ema_200": round(ema_200, 2), "above_ema": above_ema,
                "ret_3m_pct": ret_3m, "ret_6m_pct": ret_6m, "verdict": v_code,
                "verdict_text": f"🟢 HEALTHY" if v_code == "GREEN" else (f"🟡 MONITOR" if v_code == "YELLOW" else f"🔴 REVIEW"),
                "recommendation": "ACCUMULATE" if v_code == "GREEN" else ("SIP ONLY" if v_code == "YELLOW" else "PAUSE FRESH"),
                "exit_signal": exit_sig, "exit_note": exit_note,
                "methodology_note": "Evaluated on NAV 200 EMA trend + 3M/6M momentum."
            }
        except Exception as e:
            return {"error": str(e)}

    sym = ticker_str if ticker_str.endswith(".NS") or "^" in ticker_str else f"{ticker_str}.NS"
    try:
        w_df = yf.download(sym, period="2y", interval="1wk", progress=False)
        d_df = yf.download(sym, period="1y", interval="1d", progress=False)
        if w_df.empty or d_df.empty: return {"error": f"Asset '{ticker_str}' not found on NSE."}

        clean_df = pd.DataFrame({'High': w_df['High'].squeeze(), 'Low': w_df['Low'].squeeze(), 'Close': w_df['Close'].squeeze()}).dropna()
        trend, st_line, _ = compute_supertrend(clean_df)
        adx = compute_adx(clean_df)
        ltp, stop = float(clean_df['Close'].iloc[-1]), float(st_line[-1])
        ema200 = float(d_df["Close"].squeeze().ewm(span=200, adjust=False).mean().iloc[-1])
        is_bull, above_ema = (trend[-1] == 1), (ltp > ema200)
        risk_pct = round(((ltp - stop) / ltp) * 100, 2) if (is_bull and ltp > stop > 0) else 0.0

        v_code = "GREEN" if (is_bull and adx and adx >= 25.0 and above_ema) else ("YELLOW" if (is_bull or above_ema) else "RED")
        return {
            "asset_type": "stock", "ltp": round(ltp, 2), "st_bullish": is_bull, "adx": round(adx, 1) if adx else 0.0,
            "ema_200": round(ema200, 2), "above_ema": above_ema, "dynamic_stop": round(stop, 2) if is_bull else 0.0,
            "risk_to_stop": f"-{abs(risk_pct)}", "as_of": d_df.index[-1].strftime("%d-%b-%Y"), "verdict": v_code,
            "verdict_text": "🟢 MOMENTUM ENTRY" if v_code == "GREEN" else ("🟡 CONSOLIDATION" if v_code == "YELLOW" else "🔴 CAUTION"),
            "recommendation": "ENTRY CANDIDATE" if v_code == "GREEN" else ("WATCHLIST" if v_code == "YELLOW" else "AVOID ENTRY"),
            "recommendation_note": "Dual-Gate technical checks verified."
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# 8. EMPIRICAL HISTORICAL BACKTEST ENGINE
# =============================================================================
def run_historical_backtest(tickers, adx_threshold=25.0, hold_period_weeks=8, lookback_years=3):
    """
    Backtests the Dual-Gate Momentum Strategy against historical weekly data.
    Simulates signal generation, trailing ATR stop-losses, and calculates forward expectancy.
    """
    try:
        data = yf.download(tickers=tickers, period=f"{lookback_years}y", interval="1wk", group_by="ticker", progress=False, threads=True)
    except Exception as e:
        return {"error": f"Backtest data download failed: {e}"}

    trades = []
    single = len(tickers) == 1

    for t in tickers:
        df = data if single else (data[t] if t in data.columns.get_level_values(0) else pd.DataFrame())
        if df.empty or len(df) < 50:
            continue

        clean_df = pd.DataFrame({
            'High': df['High'].squeeze(), 'Low': df['Low'].squeeze(), 'Close': df['Close'].squeeze()
        }).dropna()

        if len(clean_df) < 50:
            continue

        high, low, close = clean_df['High'].values, clean_df['Low'].values, clean_df['Close'].values
        n = len(clean_df)

        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(window=10).mean().values
        hl2 = (high + low) / 2.0
        bu, bl = hl2 + (3.0 * atr), hl2 - (3.0 * atr)

        fu, fl = np.full(n, np.nan), np.full(n, np.nan)
        trend = np.zeros(n)
        start = 10
        fu[start], fl[start] = bu[start], bl[start]
        trend[start] = 1 if close[start] > fu[start] else -1

        for i in range(start + 1, n):
            fu[i] = bu[i] if (bu[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
            fl[i] = bl[i] if (bl[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
            trend[i] = ( -1 if close[i] < fl[i] else 1 ) if trend[i-1] == 1 else ( 1 if close[i] > fu[i] else -1 )

        ema200 = clean_df['Close'].ewm(span=40, adjust=False).mean().values
        adx_series = compute_adx_series(clean_df)

        for i in range(40, n - hold_period_weeks):
            is_entry = (trend[i] == 1 and trend[i-1] == -1 and adx_series[i] >= adx_threshold and close[i] > ema200[i])
            if is_entry:
                entry_price = close[i]
                exit_price = close[i + hold_period_weeks]
                stopped_out = False

                for fwd in range(1, hold_period_weeks + 1):
                    curr_idx = i + fwd
                    if low[curr_idx] <= fl[curr_idx - 1]:
                        exit_price = fl[curr_idx - 1]
                        stopped_out = True
                        break

                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    "Ticker": t.replace(".NS", ""),
                    "Entry Date": clean_df.index[i].strftime("%d-%b-%Y"),
                    "Entry (₹)": round(entry_price, 2),
                    "Exit (₹)": round(exit_price, 2),
                    "PnL (%)": round(pnl_pct, 2),
                    "Win": pnl_pct > 0,
                    "Stopped Out": stopped_out
                })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {"total_trades": 0, "win_rate": "0.0%", "avg_return_pct": "0.0%", "profit_factor": 0.0, "trades_df": trades_df}

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df["Win"]])
    win_rate = round((wins / total_trades) * 100, 1)
    avg_ret = round(trades_df["PnL (%)"].mean(), 2)
    loss_sum = abs(trades_df[trades_df["PnL (%)"] < 0]["PnL (%)"].sum())
    profit_factor = round(trades_df[trades_df["PnL (%)"] > 0]["PnL (%)"].sum() / loss_sum, 2) if loss_sum > 0 else 99.0

    return {
        "total_trades": total_trades,
        "win_rate": f"{win_rate}%",
        "avg_return_pct": f"{avg_ret}%",
        "profit_factor": profit_factor,
        "trades_df": trades_df
    }

def compute_adx_series(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    denom = (plus_di + minus_di).abs()
    dx = ((plus_di - minus_di).abs() / denom.replace(0, np.nan)) * 100
    return dx.rolling(period).mean().fillna(0).values



# =============================================================================
# 9. LIVE IPO CALENDAR (Real NSE Data — Nothing Hardcoded)
# =============================================================================
# The previous version of this feature was a hardcoded Python list of IPO
# names/dates/subscription/GMP figures typed in once. It LOOKED live (a
# spinner, "closes today" banners) but the underlying data never changed —
# once today's date moved past the hand-typed dates, it kept confidently
# showing stale "APPLY TODAY" alerts. This replaces it with a real fetch
# against NSE's own IPO data. No date, price, or status shown below is
# ever hand-typed — they're all derived live from NSE's response and
# today's actual date.
#
# Two inputs from the OLD scorecard are DELIBERATELY left out here rather
# than faked with a "best guess":
#   - Grey Market Premium (GMP): not an NSE-published figure at all. The
#     only sources are unofficial GMP-tracker sites with no stable free
#     API and uncertain scraping terms — not something to guess at.
#   - Peer P/E valuation arbitrage & RoCE-based fundamental quality: a
#     pre-listing company has no yfinance history and no listed-peer
#     mapping anywhere in this codebase. Building that would mean
#     hand-typing a peer table — the exact hardcoding problem being fixed.
# What IS shown is sourced live: issue dates, price band, lot size, issue
# size, and (when NSE's bid-detail endpoint responds) live subscription.
# The score below is intentionally small and only uses fields that are
# actually live-sourced — see build_ipo_recommendation().
#
# IMPORTANT CAVEAT for whoever runs this next: NSE's api/all-upcoming-issues
# schema is UNOFFICIAL and this dev environment has no network access to
# test a live response against it. Field names below are a best-effort
# guess at the shape NSE's own site consumes, with multiple candidate key
# names tried per field. If parsing comes up empty, get_live_ipo_calendar()
# still returns the untouched raw NSE response so the app can show it in a
# diagnostic panel instead of silently failing — check that first if the
# table looks wrong, and the field-mapping can be tightened from there.
NSE_HOME_URL = "https://www.nseindia.com/"
NSE_IPO_API_URL = "https://www.nseindia.com/api/all-upcoming-issues?category=ipo"
NSE_IPO_DETAIL_URL = "https://www.nseindia.com/api/ipo-detail"

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json",
}


def _get_nse_session():
    """
    NSE's www.nseindia.com/api/* endpoints reject cold requests — they
    require cookies set by first loading a normal page. This two-step
    handshake (warm up on the homepage, then call the API with the same
    session) is what every unofficial NSE scraper does; the archives.
    nseindia.com CSV endpoints this tool already uses for index
    constituents don't need it, but www.nseindia.com's JSON API does.
    """
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    session.get(NSE_HOME_URL, timeout=10)
    return session


def _pick(row, *keys, default=None):
    """Try several candidate field names — NSE's exact schema is unverified."""
    for k in keys:
        if isinstance(row, dict) and row.get(k) not in (None, ""):
            return row[k]
    return default


def _parse_nse_date(val):
    if not val:
        return None
    import datetime as _dt
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(str(val).strip(), fmt).date()
        except ValueError:
            continue
    return None


def get_live_ipo_calendar():
    """
    Fetches the ACTUAL current IPO calendar from NSE — no hardcoded
    issues, no hand-typed dates. Returns (issues_df, raw_rows, meta):
      - raw_rows is the untouched NSE API response, kept so the caller can
        show it in a diagnostic panel if the parsed table comes up empty
        (i.e. NSE's field names don't match what's guessed here).
      - meta always reports what actually happened (ok/error/as_of/skip
        report) — a failed fetch returns an EMPTY result with a reason,
        NEVER a fallback to stale hardcoded data. A silent fallback here
        would just recreate the "looks live, isn't" problem being fixed.
    """
    import datetime as _dt
    try:
        session = _get_nse_session()
        resp = session.get(NSE_IPO_API_URL, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame(), [], {"ok": False, "error": f"NSE IPO API returned HTTP {resp.status_code}", "as_of": None}
        payload = resp.json()
    except Exception as e:
        return pd.DataFrame(), [], {"ok": False, "error": f"{type(e).__name__}: {e}", "as_of": None}

    raw_rows = payload if isinstance(payload, list) else payload.get("data", payload.get("all_upcoming", []))
    as_of = _dt.datetime.now().strftime("%d-%b-%Y %H:%M")
    if not raw_rows:
        return pd.DataFrame(), [], {"ok": True, "error": None, "as_of": as_of, "attempted": 0, "succeeded": 0, "skipped": []}

    today = _dt.date.today()
    processed, skipped = [], []
    detail_debug = None  # captures the FIRST live per-symbol ipo-detail attempt (subscription + lot size + fresh issue)

    for row in raw_rows:
        try:
            company = _pick(row, "companyName", "company", "symbol")
            symbol = _pick(row, "symbol", "companySymbol")
            if not company:
                skipped.append((str(row)[:60], "No recognizable company-name field — NSE schema may have changed"))
                continue

            series = _pick(row, "series", "seriesName", default="")
            segment = "SME" if ("SME" in str(series).upper() or "SME" in str(company).upper()) else "Mainboard"

            start_date = _parse_nse_date(_pick(row, "issueStartDate", "startDate", "biddingStartDate"))
            end_date = _parse_nse_date(_pick(row, "issueEndDate", "endDate", "biddingEndDate"))

            if end_date:
                if start_date and today < start_date:
                    status = "UPCOMING"
                elif today <= end_date:
                    status = "OPEN"
                else:
                    status = "CLOSED"
            else:
                status = "UNKNOWN"

            days_to_close = (end_date - today).days if (end_date and status == "OPEN") else None

            price_band = _pick(row, "issuePrice", "priceBand", "price")
            issue_size = _pick(row, "issueSize", "totalIssueSize")

            # Everything below -- subscription by category AND lot size /
            # fresh-issue split -- comes from ONE endpoint: ipo-detail.
            # Confirmed live that this single response contains both the
            # category breakdown (under "activeCat.dataList", same schema
            # this tool already parses) AND a free-text "issueInfo.dataList"
            # with a real "Bid Lot" field and an "Issue Size" description
            # that spells out Fresh Issue vs Offer-For-Sale in share counts.
            # This replaces two separate per-symbol calls with one.
            lot_size = None
            fresh_issue_pct = None
            sub_overall = sub_qib = sub_nii = sub_rii = None

            if symbol and status == "OPEN":
                capture_this_one = detail_debug is None  # only capture the first attempt, to keep meta small
                try:
                    detail_resp = session.get(NSE_IPO_DETAIL_URL, params={"symbol": symbol, "series": "EQ"}, timeout=10)
                    detail_json = None
                    if detail_resp.status_code == 200:
                        try:
                            detail_json = detail_resp.json()
                        except Exception:
                            detail_json = None

                    if capture_this_one:
                        # Trim the large demand-graph/bid-history arrays before
                        # storing for diagnostics -- not used, and they bloat
                        # the capture with hundreds of price-point entries.
                        trimmed = None
                        if isinstance(detail_json, dict):
                            trimmed = {
                                k: v for k, v in detail_json.items()
                                if k not in ("demandGraph", "demandGraphALL", "demandDataNSE", "demandDataBSE")
                            }
                        detail_debug = {
                            "symbol": symbol, "url": detail_resp.url, "status_code": detail_resp.status_code,
                            "raw_json": trimmed,
                            "raw_text_snippet": detail_resp.text[:1000] if detail_json is None else None,
                            "exception": None,
                        }

                    if isinstance(detail_json, dict):
                        active_cat = detail_json.get("activeCat", {})
                        data_list = active_cat.get("dataList", []) if isinstance(active_cat, dict) else []
                        # Confirmed live: dataList[0] is a HEADER/label row
                        # (category="Category", etc.) -- skip it. Real rows
                        # form a TWO-LEVEL hierarchy via "srNo": top-level
                        # categories have a bare integer srNo ("1"=QIB,
                        # "2"=Non-Institutional, "3"=Retail, ...), while
                        # sub-breakdowns nested under them use "1(a)", "2.1",
                        # "2.1(a)" etc. Only the bare-integer rows are the
                        # actual category totals -- sub-rows (e.g. "Foreign
                        # Institutional Investors(FIIs)" under QIB) must NOT
                        # be matched as their own category.
                        top_level_rows = [
                            r for r in data_list
                            if isinstance(r, dict)
                            and str(r.get("category", "")).strip().upper() != "CATEGORY"
                            and re.fullmatch(r"\d+", str(_pick(r, "srNo", default="")).strip())
                        ]

                        def _row_times_subscribed(r):
                            # NSE already computes the multiple under
                            # "noOfTotalMeant" (confirmed live) -- prefer it.
                            # Fall back to bid/offered shares directly if
                            # it's ever blank for a given row.
                            explicit = _pick(r, "noOfTotalMeant", "noOfTimesSubscribed", "subscriptionTimes")
                            if explicit not in (None, ""):
                                try:
                                    return round(float(str(explicit).replace(",", "")), 4)
                                except (TypeError, ValueError):
                                    pass
                            bid_shares = _pick(r, "noOfSharesBid", "noOfSharesBidFor")
                            offered_shares = _pick(r, "noOfShareOffered", "noOfSharesOffered")
                            try:
                                bid_shares = float(str(bid_shares).replace(",", ""))
                                offered_shares = float(str(offered_shares).replace(",", ""))
                                if offered_shares > 0:
                                    return round(bid_shares / offered_shares, 4)
                            except (TypeError, ValueError):
                                pass
                            return None

                        def _row_shares(r):
                            """Raw (bid, offered) share counts for a row, or (None, None)."""
                            try:
                                bid_shares = float(str(_pick(r, "noOfSharesBid", "noOfSharesBidFor")).replace(",", ""))
                                offered_shares = float(str(_pick(r, "noOfShareOffered", "noOfSharesOffered")).replace(",", ""))
                                return bid_shares, offered_shares
                            except (TypeError, ValueError):
                                return None, None

                        explicit_total = None
                        summed_bid, summed_offered = 0.0, 0.0
                        have_valid_share_counts = False

                        for r in top_level_rows:
                            cat_name = str(_pick(r, "category", default="")).upper()
                            times = _row_times_subscribed(r)
                            bid_shares, offered_shares = _row_shares(r)
                            if bid_shares is not None and offered_shares is not None and offered_shares > 0:
                                summed_bid += bid_shares
                                summed_offered += offered_shares
                                have_valid_share_counts = True

                            if times is None:
                                continue
                            if "QIB" in cat_name or ("QUALIFIED" in cat_name and "INSTITUTIONAL" in cat_name):
                                if sub_qib is None:
                                    sub_qib = times
                            elif "NON" in cat_name and "INSTITUTIONAL" in cat_name:
                                if sub_nii is None:
                                    sub_nii = times
                            elif "RII" in cat_name or "RETAIL" in cat_name:
                                if sub_rii is None:
                                    sub_rii = times
                            elif "TOTAL" in cat_name:
                                if explicit_total is None:
                                    explicit_total = times

                        # "Overall" is mathematically total shares bid / total
                        # shares offered across every top-level category --
                        # this holds regardless of whether NSE's feed happens
                        # to also send an explicit "Total" row this time (it
                        # doesn't always, based on live testing). Prefer the
                        # explicit row when NSE does provide one; otherwise
                        # compute it directly so "Overall" doesn't go blank
                        # just because a label happened to be missing.
                        if explicit_total is not None:
                            sub_overall = explicit_total
                        elif have_valid_share_counts and summed_offered > 0:
                            sub_overall = round(summed_bid / summed_offered, 4)

                        # Lot size and fresh-issue split come from the
                        # free-text "issueInfo.dataList" -- a list of
                        # {"title": ..., "value": ...} pairs, NOT a clean
                        # schema. Confirmed live field names/phrasing:
                        # "Bid Lot": "84 Equity Shares and in multiples
                        # thereof", "Issue Size": "...Fresh Issue
                        # aggregating up to Rs.X million and Offer for Sale
                        # of up to Y Equity Shares...". The Fresh Issue
                        # rupee figure is a pre-pricing TARGET (final price
                        # isn't set yet), so it's not used for the % --
                        # instead Fresh % is derived purely from share
                        # counts (Total shares from the calendar endpoint
                        # minus the OFS share count parsed here), which
                        # keeps both sides of the subtraction in the same
                        # unit rather than guessing a conversion price.
                        issue_info_list = detail_json.get("issueInfo", {}).get("dataList", []) \
                            if isinstance(detail_json.get("issueInfo"), dict) else []

                        def _issue_info_value(*titles):
                            for item in issue_info_list:
                                if isinstance(item, dict) and str(item.get("title") or "").strip() in titles:
                                    return item.get("value")
                            return None

                        lot_text = _issue_info_value("Bid Lot", "Minimum Order Quantity")
                        if lot_text:
                            m = re.search(r"([\d,]+)\s*Equity Shares", str(lot_text))
                            if m:
                                try:
                                    lot_size = int(m.group(1).replace(",", ""))
                                except ValueError:
                                    lot_size = None

                        issue_structure_text = _issue_info_value("Issue Size")
                        if issue_structure_text:
                            m = re.search(r"Offer for Sale of up to\s*([\d,]+)\s*Equity Shares",
                                           str(issue_structure_text), re.IGNORECASE)
                            try:
                                if m and issue_size not in (None, 0, "0", ""):
                                    ofs_shares = int(m.group(1).replace(",", ""))
                                    total_shares = float(str(issue_size).replace(",", ""))
                                    fresh_shares = total_shares - ofs_shares
                                    if total_shares > 0 and fresh_shares >= 0:
                                        fresh_issue_pct = round((fresh_shares / total_shares) * 100, 1)
                                elif "entirely" in str(issue_structure_text).lower() and "fresh issue" in str(issue_structure_text).lower() \
                                        and "offer for sale" not in str(issue_structure_text).lower():
                                    fresh_issue_pct = 100.0
                                elif "entirely" in str(issue_structure_text).lower() and "offer for sale" in str(issue_structure_text).lower() \
                                        and "fresh issue" not in str(issue_structure_text).lower():
                                    fresh_issue_pct = 0.0
                            except (TypeError, ValueError):
                                fresh_issue_pct = None
                except Exception as e:
                    if capture_this_one:
                        detail_debug = {
                            "symbol": symbol, "url": NSE_IPO_DETAIL_URL, "status_code": None,
                            "raw_json": None, "raw_text_snippet": None,
                            "exception": f"{type(e).__name__}: {e}",
                        }
                    pass  # subscription/lot-size is best-effort; the calendar row still stands without it

            def _num(v):
                try:
                    return round(float(v), 2) if v not in (None, "") else None
                except (TypeError, ValueError):
                    return None

            processed.append({
                "Company": company,
                "Symbol": symbol or "—",
                "Segment": segment,
                "Status": status,
                "Issue Opens": start_date.strftime("%d-%b-%Y") if start_date else "—",
                "Issue Closes": end_date.strftime("%d-%b-%Y") if end_date else "—",
                "Days to Close": days_to_close,
                "Price Band": str(price_band) if price_band else "—",
                "Lot Size": lot_size if lot_size else "Not published by NSE's public feed",
                "Issue Size": issue_size or "—",
                "Fresh Issue (%)": fresh_issue_pct if fresh_issue_pct is not None else "Not published by NSE's public feed",
                "Sub — Overall (x)": _num(sub_overall),
                "Sub — QIB (x)": _num(sub_qib),
                "Sub — NII/HNI (x)": _num(sub_nii),
                "Sub — RII (x)": _num(sub_rii),
            })
        except Exception as e:
            skipped.append((str(_pick(row, "companyName", "symbol", default="unknown"))[:40], str(e)))

    df = pd.DataFrame(processed)
    meta = {
        "ok": True, "error": None, "as_of": as_of,
        "attempted": len(raw_rows), "succeeded": len(processed), "skipped": skipped,
        "detail_debug": detail_debug,
        "calendar_row_sample": raw_rows[0] if raw_rows else None,
    }
    return df, raw_rows, meta


def build_ipo_recommendation(df):
    """
    A rule-based Apply/Watch/Avoid read, built ONLY from parameters that
    are actually live-sourced in get_live_ipo_calendar() -- subscription
    by category (Overall / QIB / NII / RII) and Fresh Issue %. No GMP, no
    peer P/E, no RoCE: see the module note above for why those stay out
    rather than being estimated.

    Why these specific parameters: QIB (institutional) subscription is
    the single most literature-backed FREE, LIVE signal of post-listing
    performance -- institutions do real diligence before bidding, unlike
    retail. A wide gap between strong retail demand and weak QIB demand
    is a well-documented caution pattern (retail-driven hype without
    institutional confirmation), not something invented for this tool.
    Fresh Issue % matters because money raised via fresh shares funds the
    company's growth, while a high Offer-For-Sale share mostly cashes out
    existing promoters/investors -- a standard, non-fabricated heuristic.

    This is a RULE, not a guarantee -- it reads exactly like the
    recommendation/recommendation_note pattern already used for stocks
    and funds elsewhere in this file: what the tool's rule says, and
    why, so you can weigh it rather than follow it blindly. Rows that
    aren't OPEN yet, or are OPEN but NSE never returned a usable
    subscription figure, are marked "Calendar only" -- never scored on
    guessed numbers.
    """
    if df.empty:
        return df
    df = df.copy()

    def _reco(row):
        inputs_used = []
        if row["Status"] != "OPEN":
            return "⚪ CALENDAR ONLY", "Bidding hasn't opened yet — no live demand data to assess.", "0/2"

        qib, overall = row.get("Sub — QIB (x)"), row.get("Sub — Overall (x)")
        rii = row.get("Sub — RII (x)")
        fresh_pct = row.get("Fresh Issue (%)")
        fresh_pct = fresh_pct if isinstance(fresh_pct, (int, float)) else None  # may be a "not published" label string

        if qib is not None:
            inputs_used.append("QIB subscription")
        if fresh_pct is not None:
            inputs_used.append("Fresh Issue %")
        completeness = f"{len(inputs_used)}/2 live inputs"

        if qib is None and overall is None:
            return "⚪ CALENDAR ONLY", "Issue is open, but NSE hasn't returned a subscription figure yet — check back closer to close.", completeness

        if qib is not None:
            if qib >= 10:
                label, note = "🟢 STRONG — Institutional-Backed Demand", f"QIB subscribed {qib}x — real institutional diligence is backing this issue, historically the strongest free live signal available."
            elif qib >= 3:
                label, note = "🟡 MODERATE — Some Institutional Interest", f"QIB subscribed {qib}x — decent but not standout institutional demand."
            elif rii is not None and rii >= 5 and qib < 1:
                label, note = "🟠 CAUTION — Retail-Driven, Weak Institutional Backing", f"Retail is {rii}x subscribed but QIB is only {qib}x — a known divergence pattern where retail enthusiasm isn't confirmed by institutional diligence."
            else:
                label, note = "🔴 WEAK — Limited Institutional Interest", f"QIB subscribed only {qib}x so far."
        else:
            # QIB unavailable — fall back to overall subscription, flagged as lower-confidence
            if overall >= 10:
                label, note = "🟡 MODERATE — Strong Overall Demand", f"Overall {overall}x subscribed, but institutional (QIB) breakdown wasn't available — this reading is lower-confidence without it."
            elif overall >= 2:
                label, note = "🟡 WATCH — Building Demand", f"Overall {overall}x subscribed so far, institutional breakdown unavailable."
            else:
                label, note = "🔴 WEAK — Limited Demand So Far", f"Overall subscription only {overall}x."

        if fresh_pct is not None:
            if fresh_pct < 30:
                note += f" Fresh issue is only {fresh_pct}% of the raise — most proceeds are an existing-investor exit (OFS), not growth capital."
            else:
                note += f" Fresh issue is {fresh_pct}% of the raise, funding the company itself rather than mostly an OFS exit."

        return label, note, completeness

    tags = [_reco(r) for _, r in df.iterrows()]
    df["Recommendation"] = [t[0] for t in tags]
    df["Recommendation Note"] = [t[1] for t in tags]
    df["Live Data Completeness"] = [t[2] for t in tags]
    return df
