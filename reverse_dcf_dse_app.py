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
import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
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


@dataclass
class DSECompanyData:
    ticker: str
    url: str
    company_name: str | None = None
    sector: str | None = None
    last_traded_price: float | None = None
    audited_pe: float | None = None
    basic_eps: float | None = None
    market_cap_mn: float | None = None
    paid_up_capital_mn: float | None = None
    total_securities: float | None = None
    error: str | None = None


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


DSE_BASE_URL = "https://www.dsebd.org/displayCompany.php?name="
DSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
KNOWN_SECTOR_PEER_SEEDS = {
    "telecommunication": ["GP", "ROBI", "BSCPLC"],
    "pharmaceuticals & chemicals": [
        "SQURPHARMA",
        "BXPHARMA",
        "RENATA",
        "IBNSINA",
        "BEACONPHAR",
        "ACMELAB",
        "ORIONPHARM",
        "NAVANAPHAR",
    ],
    "bank": [
        "BRACBANK",
        "CITYBANK",
        "EBL",
        "DUTCHBANGL",
        "PRIMEBANK",
        "PUBALIBANK",
        "ISLAMIBANK",
        "UCB",
        "MTB",
    ],
    "cement": ["LHB", "CROWNCEMNT", "PREMIERCEM", "CONFIDCEM", "HEIDELBCEM", "MEGHNACEM"],
    "fuel & power": ["UPGDCL", "SUMITPOWER", "KPCL", "DOREENPWR", "DESCO", "TITASGAS", "POWERGRID"],
    "food & allied": ["OLYMPIC", "BATBC", "UNILEVERCL", "LOVELLO", "RDFOOD", "FINEFOODS"],
    "textile": ["ENVOYTEX", "SQUARETEXT", "SAIHAMCOT", "SAIHAMTEX", "MATINSPINN", "PRIMETEX"],
    "engineering": ["BSRMLTD", "BSRMSTEEL", "GPHISPAT", "WALTONHIL", "SINGERBD", "RANFOUNDRY"],
    "insurance": ["GREENDELT", "PIONEERINS", "RELIANCINS", "PRAGATIINS", "CENTRALINS", "EASTERNINS"],
}


def build_dse_company_url(ticker: str) -> str:
    return f"{DSE_BASE_URL}{ticker.strip().upper()}"


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = clean_text(value)
    if text in {"", "-", "N/A", "n/a", "nan", "None"}:
        return None
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return to_float(match.group(0))


def extract_value_after_label(text: str, label: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    target = clean_text(label).lower()
    for index, line in enumerate(lines):
        if clean_text(line).lower() == target and index + 1 < len(lines):
            return lines[index + 1]
    for index, line in enumerate(lines):
        if target in clean_text(line).lower() and index + 1 < len(lines):
            return lines[index + 1]
    return None


def extract_audited_pe(tables: list[pd.DataFrame]) -> float | None:
    pe_tables = []
    for table in tables:
        table_text = clean_text(table.to_string()).lower()
        if "current p/e ratio using basic eps" in table_text:
            pe_tables.append(table)

    if not pe_tables:
        return None

    audited_table = pe_tables[-1]
    for _, row in audited_table.iterrows():
        row_values = [clean_text(value) for value in row.tolist()]
        if row_values and "current p/e ratio using basic eps" in row_values[0].lower():
            numeric_values = [parse_number(value) for value in row_values[1:]]
            numeric_values = [value for value in numeric_values if value is not None]
            return numeric_values[-1] if numeric_values else None
    return None


def extract_basic_eps(tables: list[pd.DataFrame], last_traded_price: float | None, audited_pe: float | None) -> float | None:
    for table in tables:
        table_text = clean_text(table.to_string()).lower()
        if "financial performance as per audited" not in table_text and "eps - continuing operations" not in table_text:
            continue
        data_rows = table.iloc[3:] if len(table) > 3 else table
        for _, row in data_rows.sort_index(ascending=False).iterrows():
            values = row.tolist()
            year = parse_number(values[0] if values else None)
            if year is None:
                continue
            # DSE audited financial table commonly stores EPS continuing ops basic original in column 5.
            for idx in [5, 6, 2, 3]:
                if idx < len(values):
                    eps = parse_number(values[idx])
                    if eps is not None:
                        return eps

    if last_traded_price and audited_pe and audited_pe > 0:
        return last_traded_price / audited_pe
    return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def scrape_dse_company_data(ticker: str) -> DSECompanyData:
    symbol = ticker.strip().upper()
    url = build_dse_company_url(symbol)
    if not symbol:
        return DSECompanyData(ticker=symbol, url=url, error="Ticker is empty.")

    try:
        response = requests.get(url, headers=DSE_HEADERS, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        return DSECompanyData(ticker=symbol, url=url, error=f"DSE request failed: {exc}")

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text("\n", strip=True)
        tables = pd.read_html(BytesIO(response.content))
    except Exception as exc:
        return DSECompanyData(ticker=symbol, url=url, error=f"DSE page parse failed: {exc}")

    company_name = extract_value_after_label(page_text, "Company Name:")
    trading_code = extract_value_after_label(page_text, "Trading Code:") or symbol
    sector = extract_value_after_label(page_text, "Sector")
    last_price = parse_number(extract_value_after_label(page_text, "Last Trading Price"))
    market_cap = parse_number(extract_value_after_label(page_text, "Market Capitalization (mn)"))
    paid_up = parse_number(extract_value_after_label(page_text, "Paid-up Capital (mn)"))
    securities = parse_number(extract_value_after_label(page_text, "Total No. of Outstanding Securities"))
    audited_pe = extract_audited_pe(tables)
    basic_eps = extract_basic_eps(tables, last_price, audited_pe)

    if company_name is None and audited_pe is None and sector is None:
        return DSECompanyData(
            ticker=symbol,
            url=url,
            error="Could not find company data on the DSE page. Check the ticker.",
        )

    return DSECompanyData(
        ticker=clean_text(trading_code).upper(),
        url=url,
        company_name=company_name,
        sector=sector,
        last_traded_price=last_price,
        audited_pe=audited_pe,
        basic_eps=basic_eps,
        market_cap_mn=market_cap,
        paid_up_capital_mn=paid_up,
        total_securities=securities,
    )


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_dse_symbol_universe(seed_ticker: str = "GP") -> list[str]:
    data = scrape_dse_company_data(seed_ticker)
    if data.error:
        return []
    try:
        html = requests.get(build_dse_company_url(seed_ticker), headers=DSE_HEADERS, timeout=20).text
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")
    candidates: set[str] = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(r"displayCompany\.php\?name=([A-Z0-9&\-.()]+)", href, flags=re.I)
        if match:
            candidates.add(match.group(1).upper())
    if not candidates:
        text = soup.get_text(" ", strip=True)
        candidates.update(re.findall(r"\b[A-Z][A-Z0-9&().-]{2,14}\b(?=\s+\d+\.\d{1,2})", text))
    return sorted(candidates)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def find_same_sector_peers(target_ticker: str, target_sector: str | None, max_scan: int = 320) -> list[str]:
    if not target_sector:
        return []
    target = target_ticker.upper()
    sector_key = clean_text(target_sector).lower()
    seeded = [
        symbol
        for key, symbols in KNOWN_SECTOR_PEER_SEEDS.items()
        if key in sector_key or sector_key in key
        for symbol in symbols
    ]
    if seeded:
        peers = []
        for symbol in sorted(set(seeded)):
            if symbol == target:
                continue
            company = scrape_dse_company_data(symbol)
            if company.error:
                continue
            if clean_text(company.sector).lower() == sector_key:
                peers.append(company.ticker)
        return sorted(set(peers))

    universe = [symbol for symbol in get_dse_symbol_universe(target) if symbol != target]
    peers: list[str] = []
    for symbol in universe[:max_scan]:
        company = scrape_dse_company_data(symbol)
        if company.error:
            continue
        if clean_text(company.sector).lower() == clean_text(target_sector).lower():
            peers.append(company.ticker)
    return sorted(set(peers))


def company_data_to_dict(company: DSECompanyData) -> dict[str, Any]:
    return {
        "Ticker": company.ticker,
        "Company Name": company.company_name,
        "Sector": company.sector,
        "Market Price": company.last_traded_price,
        "Audited P/E": company.audited_pe,
        "Basic EPS": company.basic_eps,
        "Market Cap (mn)": company.market_cap_mn,
        "Error": company.error,
        "URL": company.url,
    }


def scrape_peer_company_data(peer_tickers: list[str]) -> pd.DataFrame:
    rows = [company_data_to_dict(scrape_dse_company_data(ticker)) for ticker in peer_tickers]
    return pd.DataFrame(rows)


def clean_peer_pe_data(peer_df: pd.DataFrame, outlier_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if peer_df.empty:
        return peer_df.copy(), peer_df.copy()

    cleaned = peer_df.copy()
    cleaned["Audited P/E"] = pd.to_numeric(cleaned["Audited P/E"], errors="coerce")
    cleaned["Basic EPS"] = pd.to_numeric(cleaned["Basic EPS"], errors="coerce")
    cleaned["Market Price"] = pd.to_numeric(cleaned["Market Price"], errors="coerce")
    reasons = []
    for _, row in cleaned.iterrows():
        reason = ""
        if row.get("Error"):
            reason = row["Error"]
        elif pd.isna(row["Audited P/E"]):
            reason = "Missing or non-numeric P/E"
        elif row["Audited P/E"] <= 0:
            reason = "Zero or negative P/E"
        elif row["Audited P/E"] > outlier_threshold:
            reason = f"P/E above outlier threshold ({outlier_threshold:g})"
        reasons.append(reason)
    cleaned["Exclusion Reason"] = reasons

    included = cleaned[cleaned["Exclusion Reason"] == ""].copy()
    excluded = cleaned[cleaned["Exclusion Reason"] != ""].copy()
    return included, excluded


def calculate_peer_pe_statistics(valid_peer_df: pd.DataFrame) -> dict[str, float | None]:
    pe_values = pd.to_numeric(valid_peer_df.get("Audited P/E", pd.Series(dtype=float)), errors="coerce").dropna()
    if pe_values.empty:
        return {
            "average": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "trimmed_average": None,
            "count": 0,
        }
    trimmed = pe_values.sort_values()
    if len(trimmed) >= 5:
        trim_count = max(1, int(len(trimmed) * 0.10))
        trimmed = trimmed.iloc[trim_count:-trim_count] if len(trimmed) > trim_count * 2 else trimmed
    return {
        "average": float(pe_values.mean()),
        "median": float(pe_values.median()),
        "minimum": float(pe_values.min()),
        "maximum": float(pe_values.max()),
        "trimmed_average": float(trimmed.mean()),
        "count": int(len(pe_values)),
    }


def run_relative_valuation(
    *,
    target_eps: float,
    current_price: float,
    peer_stats: dict[str, float | None],
    selected_multiple: str,
    custom_pe: float,
) -> dict[str, float | None]:
    multiple_map = {
        "Average P/E": peer_stats.get("average"),
        "Median P/E": peer_stats.get("median"),
        "Trimmed average P/E": peer_stats.get("trimmed_average"),
        "Custom P/E": custom_pe,
    }
    selected_pe = multiple_map.get(selected_multiple)
    implied_value = target_eps * selected_pe if target_eps is not None and selected_pe is not None else None
    upside = (
        (implied_value - current_price) / current_price
        if implied_value is not None and current_price and current_price > 0
        else None
    )
    return {
        "selected_pe": selected_pe,
        "implied_value": implied_value,
        "upside": upside,
    }


def build_peer_pe_chart(valid_peer_df: pd.DataFrame, target_company: DSECompanyData, peer_stats: dict[str, float | None]) -> go.Figure:
    chart_df = valid_peer_df[["Ticker", "Audited P/E"]].copy() if not valid_peer_df.empty else pd.DataFrame(columns=["Ticker", "Audited P/E"])
    if target_company.audited_pe is not None:
        chart_df = pd.concat(
            [
                pd.DataFrame([{"Ticker": f"{target_company.ticker} (Target)", "Audited P/E": target_company.audited_pe}]),
                chart_df,
            ],
            ignore_index=True,
        )
    colors = ["#fbbf24" if "Target" in ticker else "#14b8a6" for ticker in chart_df["Ticker"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["Ticker"],
            y=chart_df["Audited P/E"],
            marker_color=colors,
            hovertemplate="%{x}<br>P/E: %{y:.2f}x<extra></extra>",
        )
    )
    if peer_stats.get("average") is not None:
        fig.add_hline(y=peer_stats["average"], line_color="#60a5fa", line_dash="dash", annotation_text="Peer average")
    if peer_stats.get("median") is not None:
        fig.add_hline(y=peer_stats["median"], line_color="#fbbf24", line_dash="dot", annotation_text="Peer median")
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font={"color": "#dbe7f5"},
        xaxis_title="Company",
        yaxis_title="Audited P/E",
    )
    fig.update_xaxes(color="#dbe7f5")
    fig.update_yaxes(color="#dbe7f5", gridcolor="#263244")
    return fig


def build_pe_sensitivity_table(target_eps: float) -> pd.DataFrame:
    eps_cases = [target_eps * 0.90, target_eps, target_eps * 1.10]
    pe_cases = [8, 10, 12, 15, 20, 25]
    rows = []
    for eps in eps_cases:
        row = {"EPS Case": f"{eps:.2f}"}
        for pe in pe_cases:
            row[f"{pe}x P/E"] = eps * pe
        rows.append(row)
    return pd.DataFrame(rows)


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
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #07111f 0%, #0b1220 52%, #0d1828 100%);
                border-right: 1px solid #243044;
            }
            section[data-testid="stSidebar"] > div {
                padding-top: 1.2rem;
            }
            .nav-brand {
                background: linear-gradient(135deg, rgba(20, 184, 166, 0.20), rgba(37, 99, 235, 0.12));
                border: 1px solid rgba(94, 234, 212, 0.22);
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 14px;
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
            }
            .nav-kicker {
                color: #5eead4;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 5px;
            }
            .nav-title {
                color: #f8fafc;
                font-size: 1.18rem;
                font-weight: 850;
                line-height: 1.15;
                margin-bottom: 6px;
            }
            .nav-subtitle,
            .nav-note {
                color: #a9bad0;
                font-size: 0.84rem;
                line-height: 1.45;
            }
            .nav-note {
                border: 1px solid #243044;
                border-radius: 8px;
                padding: 12px;
                margin-top: 12px;
                background: rgba(15, 23, 42, 0.78);
            }
            section[data-testid="stSidebar"] [data-testid="stRadio"] {
                background: rgba(16, 24, 39, 0.92);
                border: 1px solid #243044;
                border-radius: 8px;
                padding: 8px;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 7px;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] label {
                min-height: 46px;
                padding: 10px 12px;
                border: 1px solid transparent;
                border-radius: 7px;
                background: #0f172a;
                transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
                background: #13243a;
                border-color: rgba(94, 234, 212, 0.42);
                transform: translateY(-1px);
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, rgba(20, 184, 166, 0.28), rgba(37, 99, 235, 0.16));
                border-color: rgba(94, 234, 212, 0.66);
                box-shadow: inset 3px 0 0 #5eead4;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] label p {
                color: #e7edf7 !important;
                font-weight: 750;
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
            .target-company-card {
                background:
                    radial-gradient(circle at top right, rgba(94, 234, 212, 0.18), transparent 30%),
                    linear-gradient(135deg, #162235 0%, #101827 100%);
                border: 1px solid rgba(94, 234, 212, 0.30);
                border-radius: 8px;
                padding: 20px 20px 18px;
                margin-bottom: 14px;
                box-shadow: 0 18px 48px rgba(0, 0, 0, 0.34);
            }
            .target-topline {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 14px;
                flex-wrap: wrap;
            }
            .target-company-name {
                color: #f8fafc;
                font-size: 1.52rem;
                line-height: 1.16;
                font-weight: 850;
                margin-bottom: 8px;
            }
            .target-company-meta {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                align-items: center;
            }
            .target-badge,
            .target-sector {
                border-radius: 7px;
                padding: 6px 9px;
                font-size: 0.80rem;
                font-weight: 800;
                letter-spacing: 0.01em;
            }
            .target-badge {
                background: rgba(94, 234, 212, 0.15);
                color: #99f6e4;
                border: 1px solid rgba(94, 234, 212, 0.32);
            }
            .target-sector {
                background: rgba(148, 163, 184, 0.12);
                color: #dbe7f5;
                border: 1px solid rgba(148, 163, 184, 0.22);
            }
            .target-link {
                color: #7dd3fc;
                text-decoration: none;
                font-size: 0.86rem;
                font-weight: 800;
                border: 1px solid rgba(125, 211, 252, 0.26);
                border-radius: 7px;
                padding: 8px 10px;
                background: rgba(14, 165, 233, 0.08);
            }
            .target-company-caption {
                color: #a9bad0;
                font-size: 0.88rem;
                line-height: 1.45;
                margin-top: 14px;
                max-width: 820px;
            }
            .target-metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-bottom: 18px;
            }
            .target-mini-card {
                background: linear-gradient(180deg, #162235 0%, #101827 100%);
                border: 1px solid #2d3a50;
                border-radius: 8px;
                padding: 15px 16px;
                min-height: 106px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
            }
            .target-mini-card.primary {
                border-color: rgba(94, 234, 212, 0.42);
                background: linear-gradient(180deg, rgba(20, 184, 166, 0.17), #101827 100%);
            }
            .target-mini-label {
                color: #9fb2c8;
                font-size: 0.79rem;
                font-weight: 800;
                margin-bottom: 9px;
            }
            .target-mini-value {
                color: #f8fafc;
                font-size: 1.28rem;
                font-weight: 850;
                line-height: 1.18;
                overflow-wrap: break-word;
            }
            .target-mini-card.primary .target-mini-value {
                color: #5eead4;
                font-size: 1.50rem;
                white-space: nowrap;
            }
            @media (max-width: 1100px) {
                .target-metric-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
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


def escape_html(value: Any) -> str:
    return html.escape("" if value is None else str(value))


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


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>
                {subtitle}
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


def render_reverse_dcf_page() -> None:
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


def render_company_summary(company: DSECompanyData | None) -> None:
    st.markdown('<div class="panel-title">Target Company</div>', unsafe_allow_html=True)
    if company is None:
        st.markdown(
            """
            <div class="empty-state">
                Enter a DSE ticker and fetch official DSE company data to begin relative valuation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    if company.error:
        st.error(company.error)
        st.caption(company.url)
        return

    company_name = escape_html(company.company_name or "Company name unavailable")
    ticker = escape_html(company.ticker)
    sector = escape_html(company.sector or "Sector unavailable")
    company_url = escape_html(company.url)
    market_cap_value = None if company.market_cap_mn is None else company.market_cap_mn * 1_000_000
    market_cap_display = "N/A"
    if market_cap_value is not None and not pd.isna(market_cap_value):
        if abs(market_cap_value) >= 10_000_000:
            market_cap_display = f"BDT {market_cap_value / 10_000_000:,.0f} crore"
        else:
            market_cap_display = format_bdt_compact(market_cap_value)
    audited_pe_display = "N/A" if company.audited_pe is None else f"{company.audited_pe:.2f}x"
    eps_display = "N/A" if company.basic_eps is None else f"BDT {company.basic_eps:,.2f}"
    price_display = format_currency(company.last_traded_price)

    st.markdown(
        f"""
        <div class="target-company-card">
            <div class="target-topline">
                <div>
                    <div class="target-company-name">{company_name}</div>
                    <div class="target-company-meta">
                        <span class="target-badge">{ticker}</span>
                        <span class="target-sector">{sector}</span>
                    </div>
                </div>
                <a class="target-link" href="{company_url}" target="_blank">Official DSE profile</a>
            </div>
            <div class="target-company-caption">
                This is the valuation anchor for the P/E comparison. Review the audited P/E and EPS quality
                before relying on peer multiples, especially when earnings are unusual or sector peers are limited.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="target-metric-grid">
            <div class="target-mini-card primary">
                <div class="target-mini-label">Audited P/E</div>
                <div class="target-mini-value">{escape_html(audited_pe_display)}</div>
            </div>
            <div class="target-mini-card">
                <div class="target-mini-label">Basic EPS</div>
                <div class="target-mini-value">{escape_html(eps_display)}</div>
            </div>
            <div class="target-mini-card">
                <div class="target-mini-label">Last traded price</div>
                <div class="target-mini-value">{escape_html(price_display)}</div>
            </div>
            <div class="target-mini-card">
                <div class="target-mini-label">Market cap</div>
                <div class="target-mini-value">{escape_html(market_cap_display)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_relative_valuation_dashboard(
    *,
    target_company: DSECompanyData | None,
    valid_peers: pd.DataFrame | None,
    excluded_peers: pd.DataFrame | None,
    peer_stats: dict[str, float | None] | None,
    valuation: dict[str, float | None] | None,
    target_eps: float | None,
    target_price: float | None,
    selected_multiple: str | None,
) -> None:
    render_company_summary(target_company)
    st.divider()

    if target_company is None or target_company.error:
        return

    if valuation is None or peer_stats is None:
        st.markdown(
            """
            <div class="empty-state">
                Select peers and run relative valuation to see peer statistics, charts, and fair value estimates.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if peer_stats.get("count", 0) < 3:
        st.warning("Too few valid peer companies. Relative valuation may be unreliable.")
    if target_eps is not None and target_eps <= 0:
        st.warning("Target EPS is zero or negative. P/E valuation is usually misleading for loss-making companies.")

    st.markdown('<div class="panel-title">Relative Valuation Output</div>', unsafe_allow_html=True)
    output_cols = st.columns(4)
    with output_cols[0]:
        render_value_card("Peer average P/E", "N/A" if peer_stats["average"] is None else f"{peer_stats['average']:.2f}x")
    with output_cols[1]:
        render_value_card("Peer median P/E", "N/A" if peer_stats["median"] is None else f"{peer_stats['median']:.2f}x", primary=True)
    with output_cols[2]:
        render_value_card("Selected multiple", "N/A" if valuation["selected_pe"] is None else f"{valuation['selected_pe']:.2f}x")
    with output_cols[3]:
        render_value_card("Implied fair value", format_currency(valuation["implied_value"]))

    upside_text = "N/A" if valuation["upside"] is None else f"{valuation['upside']:.2%}"
    st.markdown(
        f"""
        <div class="insight-note">
            Relative valuation estimates what the target may be worth if the market values it similarly
            to the selected peer group. Using <b>{selected_multiple}</b>, the implied upside/downside is
            <b>{upside_text}</b>. This is not intrinsic value; it depends heavily on peer selection and EPS quality.
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    stat_cols[0].metric("Peer count", int(peer_stats["count"]))
    stat_cols[1].metric("Min P/E", "N/A" if peer_stats["minimum"] is None else f"{peer_stats['minimum']:.2f}x")
    stat_cols[2].metric("Max P/E", "N/A" if peer_stats["maximum"] is None else f"{peer_stats['maximum']:.2f}x")
    stat_cols[3].metric("Trimmed avg P/E", "N/A" if peer_stats["trimmed_average"] is None else f"{peer_stats['trimmed_average']:.2f}x")

    st.divider()
    st.subheader("Peer P/E Comparison")
    if valid_peers is not None and not valid_peers.empty:
        st.plotly_chart(
            build_peer_pe_chart(valid_peers, target_company, peer_stats),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.dataframe(
            style_dark_dataframe(
                valid_peers.style.format(
                    {
                        "Market Price": "BDT {:,.2f}",
                        "Audited P/E": "{:.2f}x",
                        "Basic EPS": "BDT {:,.2f}",
                        "Market Cap (mn)": "{:,.2f}",
                    },
                    na_rep="N/A",
                )
            ),
            use_container_width=True,
            hide_index=True,
        )

    if excluded_peers is not None and not excluded_peers.empty:
        with st.expander("Excluded peers and reasons"):
            st.dataframe(
                style_dark_dataframe(
                    excluded_peers.style.format(
                        {
                            "Market Price": "BDT {:,.2f}",
                            "Audited P/E": "{:.2f}x",
                            "Basic EPS": "BDT {:,.2f}",
                        },
                        na_rep="N/A",
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("P/E Sensitivity")
    if target_eps is not None and target_eps > 0:
        sensitivity = build_pe_sensitivity_table(target_eps)
        st.dataframe(
            style_dark_dataframe(sensitivity.style.format({col: "BDT {:,.2f}" for col in sensitivity.columns if col != "EPS Case"})),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Export")
    export_cols = st.columns(2)
    if valid_peers is not None:
        export_cols[0].download_button(
            "Peer valuation CSV",
            data=valid_peers.to_csv(index=False).encode("utf-8"),
            file_name=f"{target_company.ticker}_pe_valid_peers.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if excluded_peers is not None:
        export_cols[1].download_button(
            "Excluded peers CSV",
            data=excluded_peers.to_csv(index=False).encode("utf-8"),
            file_name=f"{target_company.ticker}_pe_excluded_peers.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_pe_relative_valuation_page() -> None:
    for key, default in {
        "pe_target_company": None,
        "pe_peer_options": [],
        "pe_valid_peers": None,
        "pe_excluded_peers": None,
        "pe_peer_stats": None,
        "pe_valuation": None,
        "pe_target_eps": None,
        "pe_target_price": None,
        "pe_selected_multiple": None,
        "pe_fetch_message": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    input_col, output_col = st.columns([0.36, 0.64], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown('<div class="panel-title">P/E Input Panel</div>', unsafe_allow_html=True)
            target_ticker = st.text_input(
                "Target DSE ticker",
                value="GP",
                help="Ticker is used in https://www.dsebd.org/displayCompany.php?name=(ticker)",
            ).upper().strip()

            if st.button("Fetch target company data", type="primary", use_container_width=True):
                with st.spinner("Scraping target DSE company page..."):
                    target_company = scrape_dse_company_data(target_ticker)
                    st.session_state.pe_target_company = target_company
                    st.session_state.pe_valid_peers = None
                    st.session_state.pe_excluded_peers = None
                    st.session_state.pe_peer_stats = None
                    st.session_state.pe_valuation = None
                if target_company.error:
                    st.error(target_company.error)
                else:
                    st.success(f"Fetched {target_company.ticker} from DSE.")
                    with st.spinner("Finding same-sector peers from DSE pages..."):
                        peers = find_same_sector_peers(
                            target_company.ticker,
                            target_company.sector,
                            max_scan=60,
                        )
                    st.session_state.pe_peer_options = peers
                    st.session_state.pe_fetch_message = f"Fetched {target_company.ticker} from DSE."
                    if len(peers) < 3:
                        st.warning("Limited same-sector peers found. Add manual peer tickers if needed.")
                    st.rerun()

            if st.session_state.pe_fetch_message:
                st.success(st.session_state.pe_fetch_message)

            target_company: DSECompanyData | None = st.session_state.pe_target_company
            default_eps = float(target_company.basic_eps or 0.0) if target_company else 0.0
            default_price = float(target_company.last_traded_price or 0.0) if target_company else 0.0

            st.divider()
            st.subheader("Peer Selection")
            if (
                target_company is not None
                and not target_company.error
                and not st.session_state.pe_peer_options
            ):
                with st.spinner("Finding same-sector peers from DSE pages..."):
                    st.session_state.pe_peer_options = find_same_sector_peers(
                        target_company.ticker,
                        target_company.sector,
                        max_scan=60,
                    )
            peer_options = st.session_state.pe_peer_options or []
            if target_company is not None and not target_company.error and len(peer_options) < 3:
                st.warning("Limited same-sector peers found. Add manual peer tickers if needed.")
            selected_peers = st.multiselect(
                "Same-sector peer tickers",
                options=peer_options,
                default=peer_options[: min(8, len(peer_options))],
                help="Select or remove peer companies manually before running valuation.",
            )
            manual_peer_text = st.text_input(
                "Add manual peer tickers",
                value="",
                help="Comma-separated tickers, useful when the DSE sector peer scan is limited.",
            )
            manual_peers = [ticker.strip().upper() for ticker in manual_peer_text.split(",") if ticker.strip()]
            final_peers = sorted(set(selected_peers + manual_peers))

            st.divider()
            st.subheader("Manual Overrides")
            target_eps = st.number_input(
                "Target EPS",
                value=default_eps,
                step=0.10,
                format="%.3f",
                help="Audited basic EPS continuing operations when available. Override for adjusted EPS.",
            )
            target_price = st.number_input(
                "Target current market price",
                min_value=0.0,
                value=default_price,
                step=1.0,
                help="Last traded price from DSE when available. Override if needed.",
            )
            outlier_threshold = st.number_input(
                "Outlier P/E threshold",
                min_value=1.0,
                value=80.0,
                step=5.0,
                help="Peers above this P/E are excluded as possible outliers.",
            )
            selected_multiple = st.selectbox(
                "Valuation multiple",
                ["Median P/E", "Average P/E", "Trimmed average P/E", "Custom P/E"],
            )
            custom_pe = st.number_input("Custom P/E multiple", min_value=0.0, value=12.0, step=0.5)

            if st.button("Run relative valuation", type="primary", use_container_width=True):
                if target_company is None or target_company.error:
                    st.error("Fetch a valid target company first.")
                elif not final_peers:
                    st.error("Select at least one peer ticker.")
                else:
                    with st.spinner("Scraping selected peer companies and cleaning P/E data..."):
                        peer_df = scrape_peer_company_data(final_peers)
                        valid_peers, excluded_peers = clean_peer_pe_data(peer_df, outlier_threshold)
                        peer_stats = calculate_peer_pe_statistics(valid_peers)
                        valuation = run_relative_valuation(
                            target_eps=target_eps,
                            current_price=target_price,
                            peer_stats=peer_stats,
                            selected_multiple=selected_multiple,
                            custom_pe=custom_pe,
                        )
                    st.session_state.pe_valid_peers = valid_peers
                    st.session_state.pe_excluded_peers = excluded_peers
                    st.session_state.pe_peer_stats = peer_stats
                    st.session_state.pe_valuation = valuation
                    st.session_state.pe_target_eps = target_eps
                    st.session_state.pe_target_price = target_price
                    st.session_state.pe_selected_multiple = selected_multiple

    with output_col:
        with st.container(border=True):
            render_relative_valuation_dashboard(
                target_company=st.session_state.pe_target_company,
                valid_peers=st.session_state.pe_valid_peers,
                excluded_peers=st.session_state.pe_excluded_peers,
                peer_stats=st.session_state.pe_peer_stats,
                valuation=st.session_state.pe_valuation,
                target_eps=st.session_state.pe_target_eps,
                target_price=st.session_state.pe_target_price,
                selected_multiple=st.session_state.pe_selected_multiple,
            )
            with st.expander("P/E relative valuation notes"):
                st.markdown(
                    """
                    Relative valuation does not estimate intrinsic value directly. It estimates what a
                    company may be worth if valued similarly to comparable companies. The output depends
                    heavily on peer selection and scraped data quality.

                    P/E can be misleading for loss-making companies, cyclical firms, banks and financials,
                    firms with unusually low EPS, or companies with one-time gains/losses. Always review
                    the peer group and EPS quality before relying on the result.
                    """
                )


def render_sidebar_navigation() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="nav-brand">
                <div class="nav-kicker">DSE Insight</div>
                <div class="nav-title">Valuation Dashboard</div>
                <div class="nav-subtitle">Switch between intrinsic-growth analysis and market-relative multiples.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Valuation Pages",
            ["Reverse DCF", "P/E Relative Valuation"],
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="nav-note">
                Reverse DCF solves the growth priced into the market. P/E valuation compares the target
                against selected DSE peers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


def main() -> None:
    st.set_page_config(page_title="DSE Insight", layout="wide")
    inject_custom_css()
    page = render_sidebar_navigation()
    if page == "Reverse DCF":
        render_page_header(
            "Dhaka Stock Exchange Reverse DCF",
            "Estimate the free-cash-flow growth rate already implied by a DSE-listed company's market price.",
        )
        render_reverse_dcf_page()
    else:
        render_page_header(
            "DSE P/E Relative Valuation",
            "Scrape official DSE company pages, compare selected peer P/E multiples, and estimate market-relative fair value.",
        )
        render_pe_relative_valuation_page()


if __name__ == "__main__":
    main()
