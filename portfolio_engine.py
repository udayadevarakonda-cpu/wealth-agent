import yfinance as yf

FUNDS_DATA = [
    {"category": "Emergency / Overnight", "name": "Nippon India Liquid Fund (Direct-Growth)", "ticker": "0P0000XW0J.BO", "ter": "0.19%", "horizon": "Immediate (<3M)"},
    {"category": "Short-Term (<3 Years)", "name": "ICICI Prudential Arbitrage Fund (Direct-Growth)", "ticker": "0P0000XV4E.BO", "ter": "0.35%", "horizon": "1 - 3 Years"},
    {"category": "Short-Term (<3 Years)", "name": "HDFC Short Term Debt Fund (Direct-Growth)", "ticker": "0P0000XV25.BO", "ter": "0.33%", "horizon": "1 - 3 Years"},
    {"category": "Long-Term (Index / Broad)", "name": "UTI Nifty 50 Index Fund (Direct-Growth)", "ticker": "0P0000XW1O.BO", "ter": "0.18%", "horizon": "5+ Years"},
    {"category": "Long-Term (Flexi-Cap / Active)", "name": "Parag Parikh Flexi Cap Fund (Direct-Growth)", "ticker": "0P0000YWL1.BO", "ter": "0.62%", "horizon": "5+ Years"}
]

def fetch_live_fund_metrics():
    enriched_funds = []
    for item in FUNDS_DATA:
        nav = 0.0
        r_1y, r_3y, r_5y = "N/A", "N/A", "N/A"
        try:
            df = yf.download(item["ticker"], period="5y", interval="1d", progress=False)
            if not df.empty and len(df) > 10:
                close = df['Close'].squeeze()
                nav = round(float(close.iloc[-1]), 2)
                if len(close) >= 252:
                    r_1y = f"{round(((close.iloc[-1] / close.iloc[-252]) - 1) * 100, 1)}%"
                if len(close) >= 756:
                    cagr_3 = ((close.iloc[-1] / close.iloc[-756]) ** (1/3) - 1) * 100
                    r_3y = f"{round(cagr_3, 1)}%"
                if len(close) >= 1260:
                    cagr_5 = ((close.iloc[-1] / close.iloc[-1260]) ** (1/5) - 1) * 100
                    r_5y = f"{round(cagr_5, 1)}%"
        except Exception:
            pass

        enriched_funds.append({
            "Category": item["category"],
            "Fund Name": item["name"],
            "Live NAV (₹)": nav if nav > 0 else "Active",
            "1Y Return": r_1y,
            "3Y CAGR": r_3y,
            "5Y CAGR": r_5y,
            "Expense Ratio": item["ter"],
            "Horizon": item["horizon"]
        })
    return enriched_funds

def calculate_portfolio(
    liquid_cash: float,
    monthly_expenses: float,
    emergency_months: int = 6,
    risk_profile: str = "Moderate",
    short_term_pct: float = 30.0
) -> dict:
    emergency_fund = monthly_expenses * emergency_months
    investable_cash = max(0.0, liquid_cash - emergency_fund)

    if investable_cash <= 0:
        return {"error": "Total cash is less than or equal to the required emergency reserve."}

    long_term_pct = 100.0 - short_term_pct
    short_term_total = investable_cash * (short_term_pct / 100.0)
    long_term_total = investable_cash * (long_term_pct / 100.0)

    short_term_breakdown = {
        "Arbitrage Mutual Funds": round(short_term_total * 0.50, 2),
        "Short-Duration Debt Funds": round(short_term_total * 0.50, 2)
    }

    risk = risk_profile.strip().capitalize()
    if risk == "Aggressive":
        equity_weights = {"Index Funds (Nifty 50)": 0.40, "Flexi-Cap / Active Alpha": 0.60}
    elif risk == "Conservative":
        equity_weights = {"Index Funds (Nifty 50)": 0.60, "Flexi-Cap / Active Alpha": 0.40}
    else:
        equity_weights = {"Index Funds (Nifty 50)": 0.50, "Flexi-Cap / Active Alpha": 0.50}

    long_term_breakdown = {k: round(long_term_total * w, 2) for k, w in equity_weights.items()}
    stp_weekly_installment = round(long_term_total / 24, 2) if long_term_total > 0 else 0
    funds_list = fetch_live_fund_metrics()

    return {
        "total_cash": liquid_cash,
        "emergency_reserve": emergency_fund,
        "investable_cash": investable_cash,
        "short_term_total": short_term_total,
        "short_term_breakdown": short_term_breakdown,
        "long_term_total": long_term_total,
        "long_term_breakdown": long_term_breakdown,
        "stp_weekly_installment": stp_weekly_installment,
        "stp_weeks": 24,
        "funds_performance": funds_list,
        "recommended_funds_catalog": funds_list
    }
