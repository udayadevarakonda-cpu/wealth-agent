import os
from openai import OpenAI
from dotenv import load_dotenv
from portfolio_engine import calculate_portfolio

def generate_wealth_plan(
    liquid_cash: float,
    monthly_expenses: float,
    emergency_months: int = 6,
    risk_profile: str = "Moderate",
    short_term_pct: float = 30.0,
    currency_symbol: str = "₹",
    api_key: str = None
) -> dict:
    load_dotenv(override=True)

    portfolio = calculate_portfolio(
        liquid_cash=liquid_cash,
        monthly_expenses=monthly_expenses,
        emergency_months=emergency_months,
        risk_profile=risk_profile,
        short_term_pct=short_term_pct
    )

    if "error" in portfolio:
        return {"error": portfolio["error"], "portfolio": None, "advice": None}

    active_key = api_key or os.getenv("OPENAI_API_KEY")
    if not active_key or active_key.strip() == "" or "your_openai_api_key" in active_key:
        return {
            "error": "OpenAI API Key not found. Please paste it in the sidebar or save it in .env.",
            "portfolio": portfolio,
            "advice": None
        }

    client = OpenAI(api_key=active_key.strip())

    system_prompt = (
        "You are an executive fiduciary wealth manager. "
        "Your goal is to provide a complete lifecycle investment blueprint: capital allocation, "
        "specific Direct-Growth fund selection, systematic deployment (STP), and a disciplined exit strategy."
    )

    user_prompt = f"""
Build a comprehensive wealth allocation and exit blueprint based on this portfolio calculation:

- Total Liquid Cash: {currency_symbol} {portfolio['total_cash']:,.2f}
- Emergency Reserve ({emergency_months}M): {currency_symbol} {portfolio['emergency_reserve']:,.2f}
- Net Investable Capital: {currency_symbol} {portfolio['investable_cash']:,.2f}
- Risk Profile: {risk_profile}
- Short-Term Bucket (<3 Yrs): {currency_symbol} {portfolio['short_term_total']:,.2f}
- Long-Term Bucket (5+ Yrs): {currency_symbol} {portfolio['long_term_total']:,.2f}
- Systematic Transfer Plan (STP): {portfolio['stp_weeks']} weeks @ {currency_symbol} {portfolio['stp_weekly_installment']:,.2f}/week.

Use this live verified mutual funds dataset:
{portfolio.get('funds_performance', portfolio.get('recommended_funds_catalog', []))}

Please format the blueprint with these 5 clear sections:
1. **Target Fund Allocation Table** (Category, Specific Fund Name, Allocation Amount ({currency_symbol}), Expense Ratio, Investment Horizon).
2. **Short-Term Tactical Deployment Strategy** (Why these arbitrage/liquid funds were selected for capital preservation and tax efficiency).
3. **Long-Term Systematic Deployment Plan (STP)** (Step-by-step instructions on parking the long-term lump sum into arbitrage/liquid funds and executing weekly transfers into target equity funds).
4. **Deterministic Exit & Profit-Locking Strategy (Glide Path)**:
   - Specific glide path de-risking: How and when to begin systematically shifting equity profits into arbitrage/debt 18–24 months before goal maturity.
   - Long-term capital gains tax harvesting rules.
   - Exact percentage triggers to trim overweight equity positions (e.g., drift > 5%).
5. **Annual Review & Rebalancing Checklist** (Specific benchmarks to monitor and when to replace an underperforming active fund).
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return {
            "portfolio": portfolio,
            "advice": response.choices[0].message.content,
            "error": None
        }
    except Exception as e:
        return {
            "portfolio": portfolio,
            "advice": None,
            "error": f"OpenAI API Error: {str(e)}"
        }
