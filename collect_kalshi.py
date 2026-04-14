"""
Kalshi Geopolitical Contract Data Collection
=============================================
Collects historical price and volume data for geopolitical/military
prediction market contracts from Kalshi's public REST API v2.

Usage:
    python collect_kalshi.py

Output:
    data/kalshi_markets.csv     -- Market metadata
    data/kalshi_prices.csv      -- Historical price / trade data
    data/kalshi_summary.csv     -- Summary statistics

API Reference:
    https://docs.kalshi.com/
    Base URL: https://api.elections.kalshi.com/trade-api/v2
    No authentication required for public market data endpoints.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

GEOPOLITICAL_KEYWORDS = [
    "Iran", "Israel", "strike", "war", "China", "Taiwan",
    "invasion", "Russia", "Ukraine", "military", "nuclear",
    "NATO", "missile", "attack", "conflict",
]

# Kalshi statuses to query (API only allows one at a time)
MARKET_STATUSES = ["open", "closed", "settled"]

RATE_LIMIT_DELAY = 0.3
MAX_MARKETS_PER_REQUEST = 200
# Cap pages per status to avoid scanning 70,000+ sports markets
MAX_PAGES_PER_STATUS = 50  # 10,000 markets per status max


# ---------- Market Discovery ----------

def fetch_kalshi_markets(
    status: str,
    limit: int = MAX_MARKETS_PER_REQUEST,
    cursor: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Fetch a page of Kalshi markets filtered by status.

    Returns:
        Tuple of (markets list, next cursor or None).
    """
    url = f"{KALSHI_API_BASE}/markets"
    params: Dict[str, Any] = {"limit": limit, "status": status}
    if cursor:
        params["cursor"] = cursor
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("markets", []) if isinstance(data, dict) else []
        next_cursor = data.get("cursor", None) if isinstance(data, dict) else None
        # Empty cursor means no more pages
        if next_cursor == "":
            next_cursor = None
        return markets, next_cursor
    except requests.RequestException as e:
        logger.warning("Kalshi API request failed (status=%s): %s", status, e)
        return [], None


def collect_all_kalshi_markets() -> pd.DataFrame:
    """Collect all geopolitical markets from Kalshi via pagination."""
    all_markets: Dict[str, Dict] = {}
    kw_set = {kw.lower() for kw in GEOPOLITICAL_KEYWORDS}

    for status in MARKET_STATUSES:
        logger.info("Fetching Kalshi markets with status=%s ...", status)
        cursor = None
        page = 0

        while page < MAX_PAGES_PER_STATUS:
            page += 1
            markets, next_cursor = fetch_kalshi_markets(status, cursor=cursor)
            logger.info(
                "  Page %d: received %d markets (status=%s)",
                page, len(markets), status,
            )

            if not markets:
                break

            for m in markets:
                ticker = m.get("ticker", "")
                title = m.get("title", "")
                subtitle = m.get("subtitle", "")
                searchable = (title + " " + subtitle).lower()

                # Check if any geopolitical keyword matches
                matched_kw = None
                for kw in GEOPOLITICAL_KEYWORDS:
                    if kw.lower() in searchable:
                        matched_kw = kw
                        break

                if matched_kw and ticker and ticker not in all_markets:
                    all_markets[ticker] = {
                        "ticker": ticker,
                        "title": title,
                        "subtitle": subtitle,
                        "category": m.get("category", ""),
                        "status": m.get("status", ""),
                        "open_time": m.get("open_time", ""),
                        "close_time": m.get("close_time", ""),
                        "expiration_time": m.get("expiration_time", ""),
                        "yes_bid": m.get("yes_bid", None),
                        "yes_ask": m.get("yes_ask", None),
                        "volume": m.get("volume", 0),
                        "open_interest": m.get("open_interest", 0),
                        "result": m.get("result", ""),
                        "keyword_match": matched_kw,
                    }

            if not next_cursor:
                break
            cursor = next_cursor
            time.sleep(RATE_LIMIT_DELAY)

        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(list(all_markets.values()))
    logger.info("Found %d unique Kalshi geopolitical markets.", len(df))
    return df


# ---------- Historical Data ----------

def get_kalshi_candlesticks(
    ticker: str,
    period_interval: int = 1440,
) -> List[Dict[str, Any]]:
    """Fetch candlestick data for a Kalshi market ticker.

    Args:
        ticker: Market ticker (e.g., 'IRANSTRIKE-25').
        period_interval: Candle width in minutes. Valid: 1, 60, 1440.

    Returns:
        List of candlestick dicts with keys like
        'open_price', 'close_price', 'high', 'low', 'volume', etc.
    """
    # Try the live endpoint first, then historical
    endpoints = [
        f"{KALSHI_API_BASE}/market/{ticker}/candlesticks",
        f"{KALSHI_API_BASE}/historical/markets/{ticker}/candlesticks",
    ]

    for url in endpoints:
        params = {"period_interval": period_interval}
        headers = {"Accept": "application/json"}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("candlesticks", []) if isinstance(data, dict) else []
            if candles:
                return candles
        except requests.RequestException as e:
            logger.debug("Kalshi candlesticks failed for %s at %s: %s", ticker, url, e)
            continue

    return []


def collect_kalshi_histories(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Collect candlestick histories for all Kalshi markets."""
    all_rows: List[Dict] = []

    for idx, row in markets_df.iterrows():
        ticker = row["ticker"]
        logger.info(
            "Fetching Kalshi candlesticks for %s (%d/%d)",
            ticker, idx + 1, len(markets_df),
        )
        candles = get_kalshi_candlesticks(ticker)
        logger.info("  -> received %d candles for %s", len(candles), ticker)

        for entry in candles:
            all_rows.append({
                "ticker": ticker,
                "title": row["title"],
                "timestamp": (
                    entry.get("end_period_ts")
                    or entry.get("ts")
                    or entry.get("created_time", "")
                ),
                "open_price": entry.get("open", entry.get("open_price", None)),
                "close_price": entry.get("close", entry.get("close_price", None)),
                "high_price": entry.get("high", None),
                "low_price": entry.get("low", None),
                "yes_price": entry.get("yes_price", entry.get("close", None)),
                "volume": entry.get("volume", None),
            })
        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(all_rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    logger.info("Collected %d Kalshi price observations.", len(df))
    return df


# ---------- Main ----------

def main() -> None:
    """Run full Kalshi data collection pipeline."""
    logger.info("=" * 60)
    logger.info("Kalshi Geopolitical Data Collection")
    logger.info("=" * 60)

    markets_df = collect_all_kalshi_markets()
    markets_path = DATA_DIR / "kalshi_markets.csv"
    markets_df.to_csv(markets_path, index=False)
    logger.info("Saved Kalshi market metadata to %s", markets_path)

    if markets_df.empty:
        logger.warning("No Kalshi markets found. Exiting.")
        return

    prices_df = collect_kalshi_histories(markets_df)
    prices_path = DATA_DIR / "kalshi_prices.csv"
    prices_df.to_csv(prices_path, index=False)
    logger.info("Saved Kalshi price histories to %s", prices_path)

    if not prices_df.empty:
        summary = prices_df.groupby("ticker").agg(
            title=("title", "first"),
            n_obs=("yes_price", "count"),
            min_price=("yes_price", "min"),
            max_price=("yes_price", "max"),
            mean_price=("yes_price", "mean"),
            first_date=("timestamp", "min"),
            last_date=("timestamp", "max"),
        ).reset_index()
        summary_path = DATA_DIR / "kalshi_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved Kalshi summary to %s", summary_path)

    logger.info("Kalshi data collection complete.")


if __name__ == "__main__":
    main()
