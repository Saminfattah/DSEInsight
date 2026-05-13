# DSEInsight
## Project Description

DSE Valuation Dashboard is a Python-based financial valuation tool designed for companies listed on the Dhaka Stock Exchange (DSE). The project provides a simple and interactive Streamlit web application that helps investors, analysts, and finance students perform equity valuation using two approaches: Reverse Discounted Cash Flow (Reverse DCF) and P/E Relative Valuation.

The Reverse DCF module estimates the implied growth rate embedded in a company’s current market price. Instead of forecasting a company’s fair value directly, the model works backward from the current share price to identify the growth assumptions the market may already be pricing in. The app uses available market data from the bdshare library and allows users to manually input key valuation assumptions such as free cash flow, discount rate, terminal growth rate, forecast period, shares outstanding, and net debt.

The P/E Relative Valuation module allows users to value a target DSE-listed company by comparing it with selected peer companies from the same industry. The app scrapes company-level valuation data from the official DSE website, including the current P/E ratio based on the latest audited financial statements. Users can select comparable peer companies, clean or exclude unreliable data, and estimate the target company’s implied fair value using average, median, trimmed average, or custom P/E multiples.

The application is built with a professional Streamlit dashboard interface where inputs are organized on the left side and valuation outputs are displayed on the right side. It includes valuation summaries, peer comparison tables, sensitivity analysis, charts, input validation, and clear warnings for unreliable assumptions or missing data.

This project is intended for educational, analytical, and preliminary investment research purposes. It is not financial advice, and valuation results should be interpreted carefully alongside company fundamentals, industry conditions, and market risks.

## Key Features
* Reverse DCF model for DSE-listed companies
* Implied growth rate calculation based on current market price
* Data collection using the bdshare Python library
* P/E relative valuation using scraped data from dsebd.org
* Peer company selection by industry
* Average, median, trimmed average, and custom P/E valuation methods
* Manual override for EPS, market price, and valuation assumptions
* Sensitivity analysis for valuation assumptions
* Professional Streamlit dashboard UI
* Input validation and error handling
* Peer comparison table and valuation charts
* Beginner-friendly structure for finance students and analysts

## Purpose
The main purpose of this project is to make equity valuation more accessible for DSE-listed companies by combining automated data collection, manual financial assumptions, and interactive valuation models in one simple dashboard. It helps users understand what growth expectations are implied in a stock’s current price and how the stock compares with similar companies based on market valuation multiples.
