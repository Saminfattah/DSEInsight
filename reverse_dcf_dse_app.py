"""
Reverse DCF app for Dhaka Stock Exchange shares.

Run:
    streamlit run reverse_dcf_dse_app.py

Install:
    pip install streamlit pandas numpy scipy bdshare plotly
"""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class MarketData:
    symbol: str
    latest_price: float | None
    trading_data: pd.DataFrame
    historical_data: pd.DataFrame
    warnings: list[str]
    historical_start_date: date | None = None
    historical_end_date: date | None = None


@dataclass
class DCFResult:
    growth_rate: float
    value_per_share: float
    equity_value: float
    pv_forecast_cash_flows: float
    pv_terminal_value: float
    terminal_value: float
    projected_cash_flows: pd.DataFrame


def to_float(value: Any) -> float | None:
    """Convert bdshare/string numeric values into floats."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)

    cleaned = str(value).strip().replace(",", "")
    if cleaned in {"", "-", "None", "nan", "NaN"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names returned by bdshare versions."""
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    normalized.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in normalized.columns
    ]
    return normalized


def pick_latest_price(df: pd.DataFrame) -> float | None:
    """Pick the best available current market price from bdshare output."""
    if df.empty:
        return None

    price_columns = [
        "ltp",
        "last_traded_price",
        "last_trade_price",
        "last",
        "close",
        "price",
    ]
    for column in price_columns:
        if column in df.columns:
            values = df[column].dropna()
            if not values.empty:
                price = to_float(values.iloc[0])
                if price is not None and price > 0:
                    return price
    return None


def ensure_date_column(df: pd.DataFrame, fallback_date: date | None = None) -> pd.DataFrame:
    """Ensure bdshare data has a regular date column for display and export."""
    if df.empty:
        return df

    with_date = df.copy()
    if "date" in with_date.columns:
        with_date["date"] = pd.to_datetime(with_date["date"], errors="coerce").dt.date
        return with_date

    if with_date.index.name and str(with_date.index.name).lower() == "date":
        with_date = with_date.reset_index()
        with_date["date"] = pd.to_datetime(with_date["date"], errors="coerce").dt.date
        return with_date

    if fallback_date is not None:
        with_date.insert(0, "date", fallback_date)

    return with_date


def fetch_stock_data(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> MarketData:
    """
    Fetch current and historical DSE trading data using bdshare.

    bdshare can provide trading data such as symbol, LTP, high, low, close,
    change, trades, value, and volume. Financial statement inputs needed for
    DCF are not available from bdshare and are collected manually in the UI.
    """
    warnings: list[str] = []
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Please enter a DSE trading symbol.")

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    if start_date > end_date:
        raise ValueError("Historical start date cannot be after the end date.")

    try:
        from bdshare import get_current_trade_data, get_hist_data
    except ImportError as exc:
        raise ImportError(
            "bdshare is not installed. Install it with: pip install bdshare"
        ) from exc

    try:
        trading_df = normalize_columns(get_current_trade_data(symbol))
    except Exception as exc:
        raise RuntimeError(f"Could not fetch current trading data for {symbol}: {exc}")

    if trading_df.empty:
        raise ValueError(f"No current trading data returned for symbol {symbol}.")

    if "symbol" in trading_df.columns:
        exact = trading_df[trading_df["symbol"].astype(str).str.upper() == symbol]
        if not exact.empty:
            trading_df = exact
    trading_df = ensure_date_column(trading_df, end_date)

    latest_price = pick_latest_price(trading_df)
    if latest_price is None:
        warnings.append(
            "bdshare returned trading data, but no usable latest market price was found."
        )

    try:
        historical_df = normalize_columns(
            get_hist_data(start_date.isoformat(), end_date.isoformat(), symbol)
        )
        historical_df = ensure_date_column(historical_df)
    except Exception as exc:
        historical_df = pd.DataFrame()
        warnings.append(f"Historical price data could not be fetched: {exc}")

    return MarketData(
        symbol=symbol,
        latest_price=latest_price,
        trading_data=trading_df,
        historical_data=historical_df,
        warnings=warnings,
        historical_start_date=start_date,
        historical_end_date=end_date,
    )


def validate_dcf_inputs(
    market_price: float,
    current_fcf: float,
    shares_outstanding: float,
    discount_rate: float,
    terminal_growth_rate: float,
    forecast_years: int,
) -> None:
    """Validate beginner-friendly DCF assumptions."""
    if market_price <= 0:
        raise ValueError("Current market price must be greater than zero.")
    if current_fcf <= 0:
        raise ValueError("Current free cash flow must be greater than zero.")
    if shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be greater than zero.")
    if forecast_years < 1:
        raise ValueError("Forecast period must be at least 1 year.")
    if discount_rate <= terminal_growth_rate:
        raise ValueError("Discount rate must be greater than terminal growth rate.")
    if discount_rate <= -1:
        raise ValueError("Discount rate must be greater than -100%.")
    if terminal_growth_rate <= -1:
        raise ValueError("Terminal growth rate must be greater than -100%.")


def calculate_dcf_value(
    *,
    current_fcf: float,
    shares_outstanding: float,
    discount_rate: float,
    terminal_growth_rate: float,
    forecast_years: int,
    growth_rate: float,
    cash_flow_type: str,
    net_debt: float = 0.0,
    margin_of_safety: float = 0.0,
) -> DCFResult:
    """Calculate DCF value per share for a supplied annual FCF growth rate."""
    years = np.arange(1, forecast_years + 1)
    projected_fcfs = current_fcf * np.power(1 + growth_rate, years)
    discount_factors = np.power(1 + discount_rate, years)
    pv_fcfs = projected_fcfs / discount_factors

    terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    pv_terminal_value = terminal_value / discount_factors[-1]

    operating_value = float(np.sum(pv_fcfs) + pv_terminal_value)
    if cash_flow_type == "FCFF":
        equity_value = operating_value - net_debt
    else:
        equity_value = operating_value

    value_per_share = equity_value / shares_outstanding
    value_per_share_after_mos = value_per_share * (1 - margin_of_safety)

    projected_cash_flows = pd.DataFrame(
        {
            "Year": years,
            "Projected FCF": projected_fcfs,
            "PV of FCF": pv_fcfs,
        }
    )

    return DCFResult(
        growth_rate=float(growth_rate),
        value_per_share=float(value_per_share_after_mos),
        equity_value=float(equity_value * (1 - margin_of_safety)),
        pv_forecast_cash_flows=float(np.sum(pv_fcfs)),
        pv_terminal_value=float(pv_terminal_value),
        terminal_value=float(terminal_value),
        projected_cash_flows=projected_cash_flows,
    )


def solve_implied_growth_rate(
    *,
    market_price: float,
    current_fcf: float,
    shares_outstanding: float,
    discount_rate: float,
    terminal_growth_rate: float,
    forecast_years: int,
    cash_flow_type: str,
    net_debt: float = 0.0,
    margin_of_safety: float = 0.0,
    lower_bound: float = -0.95,
    upper_bound: float = 1.00,
) -> DCFResult:
    """Solve the annual growth rate that makes DCF value equal market price."""
    validate_dcf_inputs(
        market_price,
        current_fcf,
        shares_outstanding,
        discount_rate,
        terminal_growth_rate,
        forecast_years,
    )

    def value_gap(growth_rate: float) -> float:
        result = calculate_dcf_value(
            current_fcf=current_fcf,
            shares_outstanding=shares_outstanding,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            forecast_years=forecast_years,
            growth_rate=growth_rate,
            cash_flow_type=cash_flow_type,
            net_debt=net_debt,
            margin_of_safety=margin_of_safety,
        )
        return result.value_per_share - market_price

    lower_gap = value_gap(lower_bound)
    upper_gap = value_gap(upper_bound)
    if lower_gap * upper_gap > 0:
        raise RuntimeError(
            "Solver could not bracket the implied growth rate between "
            f"{lower_bound:.0%} and {upper_bound:.0%}. Try adjusting assumptions."
        )

    try:
        from scipy.optimize import brentq

        growth_rate = brentq(value_gap, lower_bound, upper_bound, maxiter=200)
    except ImportError:
        growth_rate = bisection_root(value_gap, lower_bound, upper_bound)
    except Exception as exc:
        raise RuntimeError(f"Solver failed: {exc}") from exc

    return calculate_dcf_value(
        current_fcf=current_fcf,
        shares_outstanding=shares_outstanding,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        forecast_years=forecast_years,
        growth_rate=growth_rate,
        cash_flow_type=cash_flow_type,
        net_debt=net_debt,
        margin_of_safety=margin_of_safety,
    )


def bisection_root(
    func: Any,
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 1e-7,
    max_iterations: int = 200,
) -> float:
    """Small fallback root finder if scipy is unavailable."""
    low = lower_bound
    high = upper_bound
    low_value = func(low)

    for _ in range(max_iterations):
        mid = (low + high) / 2
        mid_value = func(mid)
        if abs(mid_value) < tolerance or (high - low) / 2 < tolerance:
            return mid
        if low_value * mid_value <= 0:
            high = mid
        else:
            low = mid
            low_value = mid_value

    raise RuntimeError("Bisection solver did not converge.")


def create_sensitivity_table(
    *,
    current_fcf: float,
    shares_outstanding: float,
    forecast_years: int,
    implied_growth_rate: float,
    cash_flow_type: str,
    net_debt: float,
    margin_of_safety: float,
    discount_rate_center: float,
    terminal_growth_center: float,
) -> pd.DataFrame:
    """Create value-per-share sensitivity for discount and terminal growth rates."""
    discount_rates = [
        max(discount_rate_center - 0.02, 0.0001),
        discount_rate_center - 0.01,
        discount_rate_center,
        discount_rate_center + 0.01,
        discount_rate_center + 0.02,
    ]
    terminal_growth_rates = [
        terminal_growth_center - 0.01,
        terminal_growth_center,
        terminal_growth_center + 0.01,
    ]

    rows: list[dict[str, Any]] = []
    for discount_rate in discount_rates:
        row: dict[str, Any] = {"Discount Rate": f"{discount_rate:.1%}"}
        for terminal_growth_rate in terminal_growth_rates:
            column = f"TG {terminal_growth_rate:.1%}"
            if discount_rate <= terminal_growth_rate:
                row[column] = np.nan
                continue
            result = calculate_dcf_value(
                current_fcf=current_fcf,
                shares_outstanding=shares_outstanding,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                forecast_years=forecast_years,
                growth_rate=implied_growth_rate,
                cash_flow_type=cash_flow_type,
                net_debt=net_debt,
                margin_of_safety=margin_of_safety,
            )
            row[column] = result.value_per_share
        rows.append(row)

    return pd.DataFrame(rows).set_index("Discount Rate")


def calculate_terminal_value_contribution(result: DCFResult) -> float:
    operating_value = result.pv_forecast_cash_flows + result.pv_terminal_value
    if operating_value == 0:
        return np.nan
    return result.pv_terminal_value / operating_value


def interpret_implied_growth(growth_rate: float, terminal_contribution: float) -> str:
    if growth_rate < 0:
        growth_view = (
            "The current price implies shrinking free cash flow over the forecast period."
        )
    elif growth_rate < 0.05:
        growth_view = "The current price implies modest free-cash-flow growth."
    elif growth_rate < 0.12:
        growth_view = "The current price implies healthy free-cash-flow growth."
    elif growth_rate < 0.25:
        growth_view = (
            "The current price implies ambitious growth, so the cash-flow assumptions "
            "deserve careful checking."
        )
    else:
        growth_view = (
            "The current price implies very high growth. This may be difficult for a "
            "mature company to sustain."
        )

    if pd.isna(terminal_contribution):
        terminal_view = ""
    elif terminal_contribution > 0.80:
        terminal_view = (
            " More than 80% of the model value comes from terminal value, so the result "
            "is highly sensitive to the discount rate and terminal growth rate."
        )
    elif terminal_contribution > 0.65:
        terminal_view = (
            " Terminal value is a large part of total value, which is normal in many DCFs "
            "but still worth monitoring."
        )
    else:
        terminal_view = (
            " The valuation is less dominated by terminal value than many long-term DCFs."
        )

    return growth_view + terminal_view


def create_scenario_analysis(
    *,
    market_price: float,
    current_fcf: float,
    shares_outstanding: float,
    discount_rate: float,
    terminal_growth_rate: float,
    forecast_years: int,
    cash_flow_type: str,
    net_debt: float,
    margin_of_safety: float,
) -> pd.DataFrame:
    """Create simple Bear/Base/Bull implied-growth scenarios."""
    scenarios = [
        {
            "Scenario": "Bear",
            "FCF": current_fcf * 0.90,
            "Discount Rate": discount_rate + 0.02,
            "Terminal Growth": terminal_growth_rate - 0.01,
        },
        {
            "Scenario": "Base",
            "FCF": current_fcf,
            "Discount Rate": discount_rate,
            "Terminal Growth": terminal_growth_rate,
        },
        {
            "Scenario": "Bull",
            "FCF": current_fcf * 1.10,
            "Discount Rate": max(discount_rate - 0.01, terminal_growth_rate + 0.01),
            "Terminal Growth": terminal_growth_rate + 0.005,
        },
    ]

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            scenario_result = solve_implied_growth_rate(
                market_price=market_price,
                current_fcf=scenario["FCF"],
                shares_outstanding=shares_outstanding,
                discount_rate=scenario["Discount Rate"],
                terminal_growth_rate=scenario["Terminal Growth"],
                forecast_years=forecast_years,
                cash_flow_type=cash_flow_type,
                net_debt=net_debt,
                margin_of_safety=margin_of_safety,
            )
            rows.append(
                {
                    "Scenario": scenario["Scenario"],
                    "Current FCF": scenario["FCF"],
                    "Discount Rate": scenario["Discount Rate"],
                    "Terminal Growth": scenario["Terminal Growth"],
                    "Implied Growth": scenario_result.growth_rate,
                    "Terminal Value Contribution": calculate_terminal_value_contribution(
                        scenario_result
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Scenario": scenario["Scenario"],
                    "Current FCF": scenario["FCF"],
                    "Discount Rate": scenario["Discount Rate"],
                    "Terminal Growth": scenario["Terminal Growth"],
                    "Implied Growth": np.nan,
                    "Terminal Value Contribution": np.nan,
                    "Note": str(exc),
                }
            )

    return pd.DataFrame(rows)


def build_sensitivity_heatmap(sensitivity: pd.DataFrame) -> go.Figure:
    numeric_sensitivity = sensitivity.apply(pd.to_numeric, errors="coerce")
    fig = go.Figure(
        data=go.Heatmap(
            z=numeric_sensitivity.values,
            x=list(numeric_sensitivity.columns),
            y=list(numeric_sensitivity.index),
            colorscale=[
                [0.0, "#172554"],
                [0.5, "#0f766e"],
                [1.0, "#fbbf24"],
            ],
            colorbar={"title": "BDT/share"},
            hovertemplate=(
                "Discount rate: %{y}<br>"
                "Terminal growth: %{x}<br>"
                "Value: BDT %{z:,.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font={"color": "#dbe7f5"},
        xaxis_title="Terminal growth",
        yaxis_title="Discount rate",
    )
    fig.update_xaxes(color="#dbe7f5")
    fig.update_yaxes(color="#dbe7f5")
    return fig


def build_excel_export(
    *,
    assumptions: dict[str, Any],
    result: DCFResult,
    sensitivity: pd.DataFrame | None,
    scenario_analysis: pd.DataFrame | None,
) -> bytes | None:
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([assumptions]).to_excel(writer, sheet_name="Assumptions", index=False)
            result.projected_cash_flows.to_excel(writer, sheet_name="Projected FCF", index=False)
            if sensitivity is not None:
                sensitivity.to_excel(writer, sheet_name="Sensitivity")
            if scenario_analysis is not None:
                scenario_analysis.to_excel(writer, sheet_name="Scenarios", index=False)
        return output.getvalue()
    except Exception:
        return None


def inject_custom_css() -> None:
    """Apply a dark finance-dashboard look."""
    st.markdown(
        """
        <style>
            .stApp {
                background: #0b111d;
                color: #e7edf7;
            }
            .block-container {
                max-width: 1420px;
                padding-top: 1.5rem;
                padding-bottom: 2.2rem;
            }
            [data-testid="stHeader"] {
                background: rgba(11, 17, 29, 0.88);
                backdrop-filter: blur(10px);
            }
            .hero {
                background: linear-gradient(135deg, #13243a 0%, #0f766e 55%, #1f7a4d 100%);
                border-radius: 8px;
                color: white;
                padding: 24px 28px;
                margin-bottom: 18px;
                border: 1px solid rgba(94, 234, 212, 0.18);
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.35);
            }
            .hero h1 {
                font-size: 2rem;
                line-height: 1.15;
                margin: 0 0 8px;
                letter-spacing: 0;
            }
            .hero p {
                font-size: 0.98rem;
                margin: 0;
                color: rgba(255, 255, 255, 0.86);
                max-width: 860px;
            }
            .stApp label,
            .stApp label p,
            .stApp [data-testid="stMarkdownContainer"] p,
            .stApp [data-testid="stCaptionContainer"] {
                color: #d6deea;
            }
            .hero p {
                color: rgba(255, 255, 255, 0.86) !important;
            }
            div[data-baseweb="input"],
            div[data-baseweb="select"] > div {
                background: #121a2a;
                border-color: #334155;
                color: #eef6ff;
            }
            .stTextInput input,
            .stNumberInput input,
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #121a2a !important;
                color: #eef6ff !important;
                -webkit-text-fill-color: #eef6ff !important;
                border-color: #334155 !important;
                opacity: 1 !important;
            }
            .stNumberInput button {
                background-color: #1a2638 !important;
                color: #e7edf7 !important;
                border-color: #334155 !important;
            }
            input {
                background-color: #121a2a !important;
                color: #eef6ff !important;
                -webkit-text-fill-color: #eef6ff !important;
                opacity: 1 !important;
            }
            .panel-title {
                font-size: 0.82rem;
                font-weight: 700;
                color: #7dd3fc;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
            }
            .insight-note {
                background: #0f2d2a;
                border: 1px solid rgba(94, 234, 212, 0.34);
                border-radius: 8px;
                color: #ccfbf1;
                padding: 13px 15px;
                font-size: 0.92rem;
                line-height: 1.45;
                margin: 10px 0 18px;
            }
            .output-focus {
                background: linear-gradient(135deg, rgba(20, 184, 166, 0.16), rgba(37, 99, 235, 0.10));
                border: 1px solid rgba(94, 234, 212, 0.28);
                border-radius: 8px;
                padding: 14px;
                margin-bottom: 16px;
            }
            .result-card {
                min-height: 112px;
                background: linear-gradient(180deg, #162235 0%, #101827 100%);
                border: 1px solid #2d3a50;
                border-radius: 8px;
                padding: 16px 16px 14px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
            }
            .result-card .label {
                color: #9fb2c8;
                font-size: 0.82rem;
                font-weight: 700;
                margin-bottom: 8px;
            }
            .result-card .value {
                color: #f8fafc;
                font-size: 1.45rem;
                font-weight: 800;
                line-height: 1.18;
                overflow-wrap: anywhere;
            }
            .result-card.primary .value {
                color: #5eead4;
                font-size: 1.75rem;
            }
            .result-card .delta {
                color: #fbbf24;
                font-size: 0.86rem;
                font-weight: 700;
                margin-top: 8px;
            }
            .empty-state {
                border: 1px dashed #475569;
                border-radius: 8px;
                padding: 28px;
                background: #0f172a;
                color: #cbd5e1;
                text-align: center;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: #243044;
                box-shadow: 0 16px 45px rgba(0, 0, 0, 0.28);
                background: #101827;
            }
            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, #162235 0%, #101827 100%);
                border: 1px solid #2d3a50;
                border-radius: 8px;
                padding: 18px 18px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
            }
            div[data-testid="stMetricLabel"] p {
                color: #9fb2c8;
                font-size: 0.86rem;
                font-weight: 700;
            }
            div[data-testid="stMetricValue"] {
                color: #f8fafc;
                font-size: 1.85rem;
                line-height: 1.15;
            }
            div[data-testid="stMetricDelta"] {
                color: #5eead4;
                font-weight: 700;
            }
            .stButton > button {
                background: #14b8a6 !important;
                border-color: #14b8a6 !important;
                color: #ffffff !important;
                border-radius: 7px;
                font-weight: 700;
            }
            .stButton > button p {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
            .stButton > button:hover {
                background: #0f766e !important;
                border-color: #0f766e !important;
                color: #ffffff !important;
            }
            h1, h2, h3, h4 {
                color: #f8fafc !important;
            }
            hr {
                border-color: #243044 !important;
            }
            .stDataFrame {
                border-radius: 8px;
                overflow: hidden;
            }
            [data-testid="stExpander"] {
                background: #0f172a;
                border-color: #243044;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"BDT {value:,.2f}"


def format_bdt_compact(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 10_000_000:
        return f"BDT {value / 10_000_000:,.2f} crore"
    if abs_value >= 100_000:
        return f"BDT {value / 100_000:,.2f} lakh"
    return f"BDT {value:,.2f}"


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def style_dark_dataframe(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    return (
        styler.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#111827"),
                        ("color", "#dbe7f5"),
                        ("border-color", "#243044"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#0f172a"),
                        ("color", "#eef6ff"),
                        ("border-color", "#243044"),
                    ],
                },
            ]
        )
    )


def render_value_card(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    primary: bool = False,
) -> None:
    card_class = "result-card primary" if primary else "result-card"
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Dhaka Stock Exchange Reverse DCF</h1>
            <p>
                A clean valuation dashboard for estimating the free-cash-flow
                growth rate already implied by a DSE-listed company's market price.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_assumption_warnings(
    *,
    discount_rate: float,
    terminal_growth_rate: float,
    forecast_years: int,
    margin_of_safety: float,
    current_fcf: float,
    shares_outstanding: float,
) -> list[str]:
    warnings: list[str] = []
    if current_fcf <= 0:
        warnings.append("Free cash flow must be positive before running the model.")
    if shares_outstanding <= 0:
        warnings.append("Shares outstanding must be positive before running the model.")
    if discount_rate <= terminal_growth_rate:
        warnings.append("Discount rate must be higher than terminal growth rate.")
    if discount_rate < 0.08:
        warnings.append("Discount rate below 8% may be aggressive for many equities.")
    if discount_rate > 0.25:
        warnings.append("Discount rate above 25% is unusually high; verify the assumption.")
    if terminal_growth_rate > 0.06:
        warnings.append("Terminal growth above 6% can overstate long-run value.")
    if forecast_years > 15:
        warnings.append("Forecast periods above 15 years can create false precision.")
    if margin_of_safety > 0.50:
        warnings.append("Margin of safety above 50% is very conservative.")
    return warnings


def build_cash_flow_chart(projected_cash_flows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=projected_cash_flows["Year"],
            y=projected_cash_flows["Projected FCF"],
            name="Projected FCF",
            marker_color="#14b8a6",
            hovertemplate="Year %{x}<br>Projected FCF: BDT %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projected_cash_flows["Year"],
            y=projected_cash_flows["PV of FCF"],
            name="PV of FCF",
            mode="lines+markers",
            line={"color": "#fbbf24", "width": 3},
            marker={"size": 7, "color": "#fde68a"},
            hovertemplate="Year %{x}<br>PV of FCF: BDT %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font={"color": "#dbe7f5"},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": "#dbe7f5"}},
        xaxis_title="Forecast year",
        yaxis_title="BDT",
        hovermode="x unified",
        hoverlabel={"bgcolor": "#111827", "bordercolor": "#334155", "font_color": "#f8fafc"},
    )
    fig.update_xaxes(
        showgrid=False,
        tickmode="linear",
        dtick=1,
        color="#dbe7f5",
        linecolor="#334155",
        zerolinecolor="#334155",
    )
    fig.update_yaxes(
        gridcolor="#263244",
        tickformat=",.0f",
        color="#dbe7f5",
        linecolor="#334155",
        zerolinecolor="#334155",
    )
    return fig


def render_market_data(market_data: MarketData | None) -> None:
    st.markdown('<div class="panel-title">Market Snapshot</div>', unsafe_allow_html=True)
    if market_data is None:
        st.caption("Fetch DSE data from the input panel to populate market information.")
        return

    st.metric("Latest market price", format_currency(market_data.latest_price))
    if market_data.historical_start_date and market_data.historical_end_date:
        st.caption(
            "Historical range: "
            f"{market_data.historical_start_date.isoformat()} to "
            f"{market_data.historical_end_date.isoformat()}"
        )

    for warning in market_data.warnings:
        st.warning(warning)

    export_cols = st.columns(2)
    if not market_data.trading_data.empty:
        export_cols[0].download_button(
            "Export current bdshare data",
            data=market_data.trading_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{market_data.symbol}_bdshare_current.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if not market_data.historical_data.empty:
        start_label = (
            market_data.historical_start_date.isoformat()
            if market_data.historical_start_date
            else "start"
        )
        end_label = (
            market_data.historical_end_date.isoformat()
            if market_data.historical_end_date
            else "end"
        )
        export_cols[1].download_button(
            "Export historical bdshare data",
            data=market_data.historical_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{market_data.symbol}_bdshare_historical_{start_label}_to_{end_label}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not market_data.trading_data.empty:
        display_cols = [
            col
            for col in [
                "date",
                "symbol",
                "ltp",
                "high",
                "low",
                "close",
                "change",
                "trade",
                "value",
                "volume",
            ]
            if col in market_data.trading_data.columns
        ]
        st.dataframe(
            market_data.trading_data[display_cols] if display_cols else market_data.trading_data,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Raw bdshare data"):
        st.dataframe(market_data.trading_data, use_container_width=True)
        if not market_data.historical_data.empty:
            st.caption("Historical rows returned by bdshare")
            st.dataframe(market_data.historical_data.head(100), use_container_width=True)


def render_result_dashboard(
    *,
    result: DCFResult | None,
    market_price: float,
    sensitivity: pd.DataFrame | None,
    scenario_analysis: pd.DataFrame | None,
    export_assumptions: dict[str, Any] | None,
) -> None:
    st.markdown('<div class="panel-title">Valuation Dashboard</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown(
            """
            <div class="empty-state">
                Fetch a DSE price, review the assumptions, then run the Reverse DCF.
                Results, sensitivity analysis, and projected cash flows will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    terminal_contribution = calculate_terminal_value_contribution(result)

    st.markdown('<div class="output-focus">', unsafe_allow_html=True)
    top_metrics = st.columns(3)
    with top_metrics[0]:
        render_value_card("Market price", format_currency(market_price))
    with top_metrics[1]:
        render_value_card("Implied growth rate", format_percent(result.growth_rate), primary=True)
    with top_metrics[2]:
        render_value_card(
            "Terminal value contribution",
            format_percent(terminal_contribution),
            primary=terminal_contribution > 0.80 if not pd.isna(terminal_contribution) else False,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="insight-note">
            {interpret_implied_growth(result.growth_rate, terminal_contribution)}
            The implied growth rate is not a forecast; it is the growth expectation
            embedded in the current share price under your assumptions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    value_metrics = st.columns(3)
    with value_metrics[0]:
        render_value_card("PV forecast FCF", format_bdt_compact(result.pv_forecast_cash_flows))
    with value_metrics[1]:
        render_value_card("PV terminal value", format_bdt_compact(result.pv_terminal_value))
    with value_metrics[2]:
        render_value_card("Equity value", format_bdt_compact(result.equity_value))

    st.divider()
    st.subheader("Scenario Analysis")
    st.caption(
        "Simple Bear/Base/Bull cases adjust FCF, discount rate, and terminal growth to show how the implied growth rate changes."
    )
    if scenario_analysis is not None:
        scenario_display = scenario_analysis.copy()
        st.dataframe(
            style_dark_dataframe(
                scenario_display.style.format(
                    {
                        "Current FCF": "BDT {:,.0f}",
                        "Discount Rate": "{:.2%}",
                        "Terminal Growth": "{:.2%}",
                        "Implied Growth": "{:.2%}",
                        "Terminal Value Contribution": "{:.2%}",
                    },
                    na_rep="N/A",
                )
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Projected Free Cash Flows")
    st.plotly_chart(
        build_cash_flow_chart(result.projected_cash_flows),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.dataframe(
        style_dark_dataframe(
            result.projected_cash_flows.style.format(
                {"Projected FCF": "BDT {:,.0f}", "PV of FCF": "BDT {:,.0f}"}
            )
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Sensitivity Heatmap")
    st.caption(
        "Rows vary the discount rate; columns vary terminal growth. Heatmap values are BDT per share."
    )
    if sensitivity is not None:
        st.plotly_chart(
            build_sensitivity_heatmap(sensitivity),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        with st.expander("View sensitivity table"):
            st.dataframe(
                style_dark_dataframe(sensitivity.style.format("BDT {:,.2f}", na_rep="N/A")),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Export")
    export_cols = st.columns(4)
    export_cols[0].download_button(
        "Projected FCF CSV",
        data=result.projected_cash_flows.to_csv(index=False).encode("utf-8"),
        file_name="reverse_dcf_projected_fcf.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if sensitivity is not None:
        export_cols[1].download_button(
            "Sensitivity CSV",
            data=sensitivity.to_csv().encode("utf-8"),
            file_name="reverse_dcf_sensitivity.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if scenario_analysis is not None:
        export_cols[2].download_button(
            "Scenarios CSV",
            data=scenario_analysis.to_csv(index=False).encode("utf-8"),
            file_name="reverse_dcf_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if export_assumptions is not None:
        excel_bytes = build_excel_export(
            assumptions=export_assumptions,
            result=result,
            sensitivity=sensitivity,
            scenario_analysis=scenario_analysis,
        )
        if excel_bytes is not None:
            export_cols[3].download_button(
                "Excel Workbook",
                data=excel_bytes,
                file_name="reverse_dcf_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


def render_methodology_notes() -> None:
    with st.expander("Methodology and data notes"):
        st.markdown(
            """
            **Reverse DCF method**

            The model uses the current market price as the target value and solves for
            the annual free-cash-flow growth rate that makes DCF value per share equal
            that market price. Future cash flows are discounted to present value, and
            terminal value is calculated using the Gordon Growth Model.

            **Collected from bdshare where available**

            - DSE symbol / trading code
            - Latest traded price (`ltp`) or closest available price field
            - Trading data such as high, low, close, change, trades, value, and volume
            - Historical trading data if bdshare returns it

            **Manual inputs**

            - Free Cash Flow to Firm (FCFF) or Free Cash Flow to Equity (FCFE)
            - Shares outstanding
            - Discount rate / WACC / cost of equity
            - Terminal growth rate
            - Forecast period
            - Net debt or cash adjustment for FCFF
            - Margin of safety
            """
        )


def main() -> None:
    st.set_page_config(page_title="DSE Insight", layout="wide")
    inject_custom_css()
    render_page_header()

    if "market_data" not in st.session_state:
        st.session_state.market_data = None
    if "market_price_input" not in st.session_state:
        st.session_state.market_price_input = 0.0
    if "dcf_result" not in st.session_state:
        st.session_state.dcf_result = None
    if "sensitivity" not in st.session_state:
        st.session_state.sensitivity = None
    if "scenario_analysis" not in st.session_state:
        st.session_state.scenario_analysis = None
    if "export_assumptions" not in st.session_state:
        st.session_state.export_assumptions = None
    if "result_market_price" not in st.session_state:
        st.session_state.result_market_price = 0.0
    if "fetch_message" not in st.session_state:
        st.session_state.fetch_message = ""

    market_data: MarketData | None = st.session_state.market_data

    input_col, output_col = st.columns([0.36, 0.64], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Input Panel</div>', unsafe_allow_html=True)
            st.subheader("Company and Market")
            ticker = st.text_input(
                "DSE ticker / trading code",
                value="GP",
                help="Enter the DSE trading symbol, for example GP, SQURPHARMA, BATBC.",
            ).upper().strip()

            with st.expander("Market data options", expanded=False):
                default_end_date = date.today()
                default_start_date = default_end_date - timedelta(days=365)
                historical_start_date = st.date_input(
                    "Historical start date",
                    value=default_start_date,
                    max_value=default_end_date,
                    help="Start date for bdshare historical price data.",
                )
                historical_end_date = st.date_input(
                    "Historical end date",
                    value=default_end_date,
                    max_value=default_end_date,
                    help="End date for bdshare historical price data.",
                )

            if st.button("Fetch DSE Data", type="primary", use_container_width=True):
                try:
                    with st.spinner("Fetching market data from bdshare..."):
                        st.session_state.market_data = fetch_stock_data(
                            ticker,
                            historical_start_date,
                            historical_end_date,
                        )
                    if st.session_state.market_data.latest_price is not None:
                        st.session_state.market_price_input = float(
                            st.session_state.market_data.latest_price
                        )
                    market_data = st.session_state.market_data
                    st.session_state.fetch_message = f"Fetched data for {ticker}."
                    st.rerun()
                except Exception as exc:
                    st.session_state.market_data = None
                    market_data = None
                    st.session_state.fetch_message = ""
                    st.error(str(exc))

            if st.session_state.fetch_message:
                st.success(st.session_state.fetch_message)

            market_price = st.number_input(
                "Current market price per share (BDT)",
                min_value=0.0,
                step=1.0,
                key="market_price_input",
                help="Fetched from bdshare when available. You can edit this manually.",
            )

            st.divider()
            st.subheader("DCF Assumptions")
            cash_flow_type = st.selectbox(
                "Cash flow basis",
                ["FCFE", "FCFF"],
                help="FCFE values equity directly. FCFF values the firm first, then subtracts net debt.",
            )
            current_fcf = st.number_input(
                f"Current {cash_flow_type} (BDT, total company cash flow)",
                min_value=0.0,
                value=1_000_000_000.0,
                step=10_000_000.0,
                help="Use trailing twelve-month or normalized free cash flow for the whole company.",
            )
            shares_outstanding = st.number_input(
                "Shares outstanding",
                min_value=0.0,
                value=1_000_000_000.0,
                step=1_000_000.0,
                help="Total ordinary shares outstanding, not market capitalization.",
            )
            discount_rate = st.number_input(
                "Discount rate / WACC / cost of equity",
                min_value=0.0,
                max_value=1.0,
                value=0.12,
                step=0.005,
                format="%.3f",
                help="Use WACC for FCFF or cost of equity for FCFE.",
            )
            terminal_growth_rate = st.number_input(
                "Terminal growth rate",
                min_value=-0.50,
                max_value=0.50,
                value=0.04,
                step=0.005,
                format="%.3f",
                help="Long-run sustainable growth rate after the explicit forecast period.",
            )
            forecast_years = st.number_input(
                "Forecast period (years)",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
            )

            with st.expander("Advanced assumptions", expanded=True):
                net_debt = st.number_input(
                    "Net debt / cash adjustment (BDT)",
                    value=0.0,
                    step=10_000_000.0,
                    help="Use positive net debt for FCFF. Use negative value for net cash. Ignored for FCFE.",
                )
                margin_of_safety = st.number_input(
                    "Margin of safety",
                    min_value=0.0,
                    max_value=0.90,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    help="Optional haircut applied to the model value.",
                )

            if cash_flow_type == "FCFE":
                net_debt = 0.0

            assumption_warnings = get_assumption_warnings(
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                forecast_years=int(forecast_years),
                margin_of_safety=margin_of_safety,
                current_fcf=current_fcf,
                shares_outstanding=shares_outstanding,
            )
            for warning in assumption_warnings:
                st.warning(warning)

            run_model = st.button("Run Reverse DCF", type="primary", use_container_width=True)

    with output_col:
        with st.container(border=True):
            render_market_data(market_data)
            st.divider()

        if run_model:
            try:
                effective_market_price = float(st.session_state.market_price_input)
                result = solve_implied_growth_rate(
                    market_price=effective_market_price,
                    current_fcf=current_fcf,
                    shares_outstanding=shares_outstanding,
                    discount_rate=discount_rate,
                    terminal_growth_rate=terminal_growth_rate,
                    forecast_years=int(forecast_years),
                    cash_flow_type=cash_flow_type,
                    net_debt=net_debt,
                    margin_of_safety=margin_of_safety,
                )
                sensitivity = create_sensitivity_table(
                    current_fcf=current_fcf,
                    shares_outstanding=shares_outstanding,
                    forecast_years=int(forecast_years),
                    implied_growth_rate=result.growth_rate,
                    cash_flow_type=cash_flow_type,
                    net_debt=net_debt,
                    margin_of_safety=margin_of_safety,
                    discount_rate_center=discount_rate,
                    terminal_growth_center=terminal_growth_rate,
                )
                scenario_analysis = create_scenario_analysis(
                    market_price=effective_market_price,
                    current_fcf=current_fcf,
                    shares_outstanding=shares_outstanding,
                    discount_rate=discount_rate,
                    terminal_growth_rate=terminal_growth_rate,
                    forecast_years=int(forecast_years),
                    cash_flow_type=cash_flow_type,
                    net_debt=net_debt,
                    margin_of_safety=margin_of_safety,
                )
                export_assumptions = {
                    "Ticker": ticker,
                    "Cash Flow Type": cash_flow_type,
                    "Market Price": effective_market_price,
                    "Current FCF": current_fcf,
                    "Shares Outstanding": shares_outstanding,
                    "Discount Rate": discount_rate,
                    "Terminal Growth Rate": terminal_growth_rate,
                    "Forecast Years": int(forecast_years),
                    "Net Debt": net_debt,
                    "Margin of Safety": margin_of_safety,
                    "Implied Growth Rate": result.growth_rate,
                    "Terminal Value Contribution": calculate_terminal_value_contribution(result),
                    "PV Forecast FCF": result.pv_forecast_cash_flows,
                    "PV Terminal Value": result.pv_terminal_value,
                    "Equity Value": result.equity_value,
                }
                st.session_state.dcf_result = result
                st.session_state.sensitivity = sensitivity
                st.session_state.scenario_analysis = scenario_analysis
                st.session_state.export_assumptions = export_assumptions
                st.session_state.result_market_price = effective_market_price
            except Exception as exc:
                st.session_state.dcf_result = None
                st.session_state.sensitivity = None
                st.session_state.scenario_analysis = None
                st.session_state.export_assumptions = None
                st.error(str(exc))

        with st.container(border=True):
            render_result_dashboard(
                result=st.session_state.dcf_result,
                market_price=st.session_state.result_market_price or market_price,
                sensitivity=st.session_state.sensitivity,
                scenario_analysis=st.session_state.scenario_analysis,
                export_assumptions=st.session_state.export_assumptions,
            )
            render_methodology_notes()


if __name__ == "__main__":
    main()
