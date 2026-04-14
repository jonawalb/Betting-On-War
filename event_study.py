"""
Event Study Analysis for Betting on War
========================================
Runs the core empirical analysis:
  1. Constructs event timeline from hand-coded geopolitical events
  2. Merges with prediction market price data
  3. Estimates event study regressions (log-odds specification)
  4. Generates event study plots and robustness checks

Usage:
    python event_study.py

Output:
    results/event_study_coefficients.csv
    results/event_study_plot.pdf
    results/liquidity_heterogeneity.csv
    results/placebo_test.csv
    results/summary_statistics.tex
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. HAND-CODED EVENT TIMELINE
# ============================================================

GEOPOLITICAL_EVENTS = [
    # Format: (date, event_type, actor, severity, description)
    # --- 2022 ---
    ("2022-02-24", "invasion", "Russia", "high",
     "Russia launches full-scale invasion of Ukraine"),
    ("2022-09-21", "mobilization", "Russia", "high",
     "Putin announces partial mobilization"),
    ("2022-10-08", "attack", "Ukraine", "medium",
     "Kerch Bridge explosion"),
    # --- 2023 ---
    ("2023-06-24", "crisis", "Russia", "high",
     "Wagner Group mutiny and march on Moscow"),
    ("2023-10-07", "attack", "Hamas", "high",
     "Hamas attacks Israel (October 7)"),
    ("2023-10-27", "military_operation", "Israel", "high",
     "Israel begins Gaza ground invasion"),
    # --- 2024 ---
    ("2024-01-12", "strike", "US/UK", "medium",
     "US and UK strike Houthi targets in Yemen"),
    ("2024-04-01", "strike", "Israel", "high",
     "Israel strikes Iranian consulate in Damascus"),
    ("2024-04-13", "retaliation", "Iran", "high",
     "Iran launches 300+ drones and missiles at Israel (True Promise 1)"),
    ("2024-04-19", "strike", "Israel", "medium",
     "Israel retaliatory strike on Isfahan"),
    ("2024-10-01", "retaliation", "Iran", "high",
     "Iran launches 180+ ballistic missiles at Israel (True Promise 2)"),
    ("2024-10-26", "strike", "Israel", "high",
     "Israel strikes Iranian military targets"),
    # --- 2025 ---
    ("2025-06-13", "strike", "US/Israel", "high",
     "Joint US-Israeli strikes on Iranian nuclear facilities"),
    ("2025-06-25", "retaliation", "Iran", "high",
     "Iranian missile barrage on Israel (True Promise 3)"),
    # --- 2026 ---
    ("2026-02-28", "strike", "US/Israel", "high",
     "Surprise joint US-Israeli attack on Iran begins current war"),
    ("2026-03-15", "escalation", "US/Israel", "high",
     "Strikes on Tehran, Isfahan, Shiraz, Mashhad"),
    ("2026-04-04", "retaliation", "Iran", "high",
     "Operation True Promise 4 -- cluster munitions on Haifa, Tel Aviv"),
]


def build_event_df() -> pd.DataFrame:
    """Convert hand-coded events into a DataFrame."""
    records = []
    for date_str, etype, actor, severity, desc in GEOPOLITICAL_EVENTS:
        records.append({
            "event_date": pd.Timestamp(date_str),
            "event_type": etype,
            "actor": actor,
            "severity": severity,
            "description": desc,
        })
    df = pd.DataFrame(records)
    df = df.sort_values("event_date").reset_index(drop=True)
    logger.info("Built event timeline with %d events.", len(df))
    return df


# ============================================================
# 2. PRICE DATA LOADING AND TRANSFORMATION
# ============================================================

def load_market_data() -> pd.DataFrame:
    """Load and concatenate Polymarket and Kalshi price data."""
    frames: List[pd.DataFrame] = []

    # Polymarket
    poly_path = DATA_DIR / "polymarket_prices.csv"
    if poly_path.exists():
        poly = pd.read_csv(poly_path, parse_dates=["timestamp"])
        poly["source"] = "polymarket"
        poly = poly.rename(columns={"price": "implied_prob"})
        if "volume" not in poly.columns:
            poly["volume"] = 0
        # Keep only "Yes" outcome rows to avoid double-counting
        if "outcome" in poly.columns:
            poly = poly[poly["outcome"] == "Yes"]
        frames.append(poly[["market_id", "timestamp", "implied_prob",
                            "volume", "question", "source"]])
        logger.info("Loaded %d Polymarket observations.", len(poly))

    # Kalshi
    kalshi_path = DATA_DIR / "kalshi_prices.csv"
    if kalshi_path.exists() and kalshi_path.stat().st_size > 10:
        kalshi = pd.read_csv(kalshi_path, parse_dates=["timestamp"])
        if not kalshi.empty:
            kalshi["source"] = "kalshi"
            kalshi = kalshi.rename(columns={
                "ticker": "market_id",
                "yes_price": "implied_prob",
                "title": "question",
            })
            # Kalshi prices are in cents; convert to probability
            if "implied_prob" in kalshi.columns and kalshi["implied_prob"].max() > 1:
                kalshi["implied_prob"] = kalshi["implied_prob"] / 100.0
            if "volume" not in kalshi.columns:
                kalshi["volume"] = 0
            frames.append(kalshi[["market_id", "timestamp", "implied_prob",
                                  "volume", "question", "source"]])
            logger.info("Loaded %d Kalshi observations.", len(kalshi))

    if not frames:
        logger.warning("No market data found. Generating synthetic data.")
        return _generate_synthetic_data()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp", "implied_prob"])
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    return df


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic prediction market data for demonstration."""
    np.random.seed(42)
    markets = [
        ("IRAN_STRIKE_2024Q4", "Israel strikes Iran by Dec 2024"),
        ("IRAN_STRIKE_2025H1", "Israel strikes Iran by June 2025"),
        ("IRAN_RETALIATION", "Iran retaliates against Israel"),
        ("US_IRAN_WAR", "US enters war with Iran by 2026"),
        ("TAIWAN_INVASION", "China invades Taiwan by 2027"),
        ("UKRAINE_CEASEFIRE", "Ukraine-Russia ceasefire by 2025"),
    ]
    rows = []
    for mid, question in markets:
        base_prob = np.random.uniform(0.05, 0.30)
        dates = pd.date_range("2024-01-01", "2026-03-31", freq="D")
        prob = base_prob
        for d in dates:
            # Random walk with mean reversion
            shock = np.random.normal(0, 0.015)
            prob = np.clip(prob + shock + 0.001 * (base_prob - prob), 0.01, 0.99)
            # Insert event-driven spikes
            for ev_date, _, _, sev, _ in GEOPOLITICAL_EVENTS:
                ev_dt = pd.Timestamp(ev_date)
                if d == ev_dt and sev == "high":
                    prob = min(prob + np.random.uniform(0.05, 0.20), 0.95)
            vol = max(0, int(np.random.exponential(50000) * prob))
            rows.append({
                "market_id": mid,
                "timestamp": d,
                "implied_prob": round(prob, 4),
                "volume": vol,
                "question": question,
                "source": "synthetic",
            })
    df = pd.DataFrame(rows)
    synth_path = DATA_DIR / "synthetic_prices.csv"
    df.to_csv(synth_path, index=False)
    logger.info("Generated %d synthetic observations.", len(df))
    return df


def to_log_odds(p: pd.Series) -> pd.Series:
    """Transform probability to log-odds for regression."""
    p_clipped = p.clip(0.001, 0.999)
    return np.log(p_clipped / (1 - p_clipped))


# ============================================================
# 3. EVENT STUDY ESTIMATION
# ============================================================

def create_event_windows(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre_window: int = 5,
    post_window: int = 10,
) -> pd.DataFrame:
    """Create event-window panel by matching events to market data."""
    prices = prices.copy()
    prices["date"] = prices["timestamp"].dt.date

    results = []
    for _, event in events.iterrows():
        ev_date = event["event_date"].date()
        window_start = ev_date - timedelta(days=pre_window)
        window_end = ev_date + timedelta(days=post_window)

        for mid in prices["market_id"].unique():
            mkt = prices[prices["market_id"] == mid].copy()
            mkt_window = mkt[
                (mkt["date"] >= window_start) & (mkt["date"] <= window_end)
            ].copy()

            if len(mkt_window) < 3:
                continue

            mkt_window["event_date"] = event["event_date"]
            mkt_window["event_type"] = event["event_type"]
            mkt_window["severity"] = event["severity"]
            mkt_window["description"] = event["description"]
            mkt_window["relative_day"] = (
                mkt_window["timestamp"].dt.date - ev_date
            ).apply(lambda x: x.days)
            mkt_window["log_odds"] = to_log_odds(mkt_window["implied_prob"])

            results.append(mkt_window)

    if results:
        panel = pd.concat(results, ignore_index=True)
        logger.info("Created event-window panel with %d obs.", len(panel))
        return panel
    else:
        logger.warning("No event windows created.")
        return pd.DataFrame()


def estimate_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate event study regression with lead/lag indicators."""
    if panel.empty:
        return pd.DataFrame()

    panel = panel.copy()
    pre_window = 5
    post_window = 10

    # Create lead/lag dummies
    for k in range(-pre_window, post_window + 1):
        if k == -1:  # reference period
            continue
        panel[f"tau_{k}"] = (panel["relative_day"] == k).astype(int)

    # Dependent variable: change in log-odds from pre-event mean
    panel["y"] = panel["log_odds"]

    # Explanatory variables
    tau_cols = [c for c in panel.columns if c.startswith("tau_")]
    X = panel[tau_cols].copy()
    X = sm.add_constant(X)
    y = panel["y"]

    # Drop missing
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(y) < len(tau_cols) + 5:
        logger.warning("Insufficient observations for event study regression.")
        return pd.DataFrame()

    model = sm.OLS(y, X).fit(cov_type="HC1")
    logger.info("Event study R-squared: %.4f", model.rsquared)

    # Extract coefficients
    coefs = []
    for k in range(-pre_window, post_window + 1):
        if k == -1:
            coefs.append({"tau": k, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
            continue
        col = f"tau_{k}"
        if col in model.params.index:
            b = model.params[col]
            se = model.bse[col]
            coefs.append({
                "tau": k,
                "coef": b,
                "se": se,
                "ci_lo": b - 1.96 * se,
                "ci_hi": b + 1.96 * se,
            })

    coef_df = pd.DataFrame(coefs).sort_values("tau")
    return coef_df


def plot_event_study(coef_df: pd.DataFrame) -> None:
    """Generate event study coefficient plot."""
    if coef_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        coef_df["tau"], coef_df["ci_lo"], coef_df["ci_hi"],
        alpha=0.2, color="steelblue",
    )
    ax.plot(coef_df["tau"], coef_df["coef"], "o-", color="steelblue",
            markersize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
               label="Event day")
    ax.set_xlabel("Days relative to event", fontsize=12)
    ax.set_ylabel("Coefficient (log-odds)", fontsize=12)
    ax.set_title("Event Study: Crisis Events and Prediction Market Prices",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "event_study_plot.pdf"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved event study plot to %s", plot_path)


# ============================================================
# 4. HETEROGENEITY BY LIQUIDITY
# ============================================================

def liquidity_heterogeneity(
    panel: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether effects are stronger in thin markets."""
    if panel.empty:
        return pd.DataFrame()

    # Compute median volume per market
    med_vol = prices.groupby("market_id")["volume"].median().reset_index()
    med_vol.columns = ["market_id", "median_volume"]
    threshold = med_vol["median_volume"].median()
    med_vol["thin_market"] = (med_vol["median_volume"] < threshold).astype(int)

    panel = panel.merge(med_vol[["market_id", "thin_market"]], on="market_id",
                        how="left")

    results = []
    for thin_val, label in [(1, "thin"), (0, "thick")]:
        sub = panel[panel["thin_market"] == thin_val]
        if len(sub) < 20:
            continue
        coefs = estimate_event_study(sub)
        if not coefs.empty:
            coefs["market_type"] = label
            results.append(coefs)

    if results:
        het_df = pd.concat(results, ignore_index=True)
        het_path = RESULTS_DIR / "liquidity_heterogeneity.csv"
        het_df.to_csv(het_path, index=False)
        logger.info("Saved liquidity heterogeneity results to %s", het_path)
        return het_df
    return pd.DataFrame()


# ============================================================
# 5. PLACEBO TESTS
# ============================================================

def placebo_test(
    prices: pd.DataFrame,
    n_placebo: int = 100,
    pre_window: int = 5,
    post_window: int = 10,
) -> pd.DataFrame:
    """Run placebo event study with random timestamps."""
    np.random.seed(123)
    date_range = prices["timestamp"].dt.date.unique()
    if len(date_range) < pre_window + post_window + 10:
        logger.warning("Insufficient date range for placebo test.")
        return pd.DataFrame()

    valid_dates = date_range[pre_window:-post_window]
    placebo_dates = np.random.choice(valid_dates, size=n_placebo, replace=True)

    placebo_events = pd.DataFrame({
        "event_date": pd.to_datetime(placebo_dates),
        "event_type": "placebo",
        "actor": "none",
        "severity": "none",
        "description": "placebo event",
    })

    panel = create_event_windows(prices, placebo_events, pre_window, post_window)
    coefs = estimate_event_study(panel)

    if not coefs.empty:
        placebo_path = RESULTS_DIR / "placebo_test.csv"
        coefs.to_csv(placebo_path, index=False)
        logger.info("Saved placebo test results to %s", placebo_path)

    return coefs


# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

def generate_summary_table(
    prices: pd.DataFrame, events: pd.DataFrame,
) -> str:
    """Generate LaTeX summary statistics table."""
    n_markets = prices["market_id"].nunique()
    n_obs = len(prices)
    date_min = prices["timestamp"].min()
    date_max = prices["timestamp"].max()
    mean_prob = prices["implied_prob"].mean()
    std_prob = prices["implied_prob"].std()
    mean_vol = prices["volume"].mean()

    tex = r"""
\begin{table}[htbp]
\centering
\caption{Summary Statistics: Prediction Market Data}
\label{tab:summary}
\begin{tabular}{lc}
\hline\hline
& \\
Number of markets & """ + str(n_markets) + r""" \\
Number of observations & """ + f"{n_obs:,}" + r""" \\
Date range & """ + f"{date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}" + r""" \\
Mean implied probability & """ + f"{mean_prob:.3f}" + r""" \\
SD implied probability & """ + f"{std_prob:.3f}" + r""" \\
Mean daily volume (\$) & """ + f"{mean_vol:,.0f}" + r""" \\
Number of crisis events & """ + str(len(events)) + r""" \\
\hline\hline
\end{tabular}
\end{table}
"""
    tex_path = RESULTS_DIR / "summary_statistics.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    logger.info("Saved summary statistics table to %s", tex_path)
    return tex


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run full event study analysis pipeline."""
    logger.info("=" * 60)
    logger.info("Event Study Analysis: Betting on War")
    logger.info("=" * 60)

    # Build event timeline
    events = build_event_df()
    events.to_csv(DATA_DIR / "event_timeline.csv", index=False)

    # Load market data
    prices = load_market_data()

    # Summary statistics
    generate_summary_table(prices, events)

    # Create event windows and estimate
    panel = create_event_windows(prices, events)
    coefs = estimate_event_study(panel)

    if not coefs.empty:
        coefs.to_csv(RESULTS_DIR / "event_study_coefficients.csv", index=False)
        plot_event_study(coefs)

    # Heterogeneity by liquidity
    liquidity_heterogeneity(panel, prices)

    # Placebo tests
    placebo_test(prices)

    logger.info("=" * 60)
    logger.info("Event study analysis complete.")
    logger.info("Results saved to: %s", RESULTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
