"""
Polymarket Geopolitical Contract Data Collection
=================================================
Collects historical price and volume data for geopolitical/military
prediction market contracts from Polymarket's Gamma and CLOB APIs.

Usage:
    python3 collect_polymarket.py

Output:
    data/polymarket_markets.csv     -- Market metadata
    data/polymarket_prices.csv      -- Historical price time series
    data/polymarket_summary.csv     -- Summary statistics

API Notes:
    - Gamma API (gamma-api.polymarket.com): Market discovery and metadata.
      Uses offset-based pagination with limit/offset params.
    - CLOB API (clob.polymarket.com): Price history via /prices-history.
      Requires CLOB token IDs (long numeric strings), not Gamma market IDs.
    - Old markets (pre-2023) may lack CLOB price history even if they
      have CLOB token IDs, since they predate the CLOB migration.
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

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

GEOPOLITICAL_KEYWORDS = [
    "iran", "israel", "russia", "ukraine", "china", "taiwan",
    "war", "military", "nuclear", "strike", "invasion", "nato",
    "missile", "attack", "bomb", "retaliation", "troops", "conflict",
    "escalation", "ceasefire", "houthi", "yemen", "gaza", "hamas",
    "hezbollah", "syria", "korea", "deterrence", "sanctions",
    "airstrike", "drone",
]

RATE_LIMIT_DELAY = 0.3  # seconds between API calls
MAX_PAGES = 60  # 6,000 markets max scan
PAGE_SIZE = 100


# ---------- Gamma API: Market Discovery ----------

def _extract_clob_token_ids(market: Dict[str, Any]) -> List[str]:
    """Extract CLOB token IDs from a Gamma API market response."""
    token_ids: List[str] = []

    # Prefer clobTokenIds (direct ID strings)
    clob_raw = market.get("clobTokenIds")
    if clob_raw:
        if isinstance(clob_raw, str):
            try:
                clob_raw = json.loads(clob_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(clob_raw, list):
            for item in clob_raw:
                if isinstance(item, str) and len(item) > 10:
                    token_ids.append(item)

    # Fallback: tokens array of objects
    if not token_ids:
        tokens_raw = market.get("tokens")
        if tokens_raw:
            if isinstance(tokens_raw, str):
                try:
                    tokens_raw = json.loads(tokens_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(tokens_raw, list):
                for t in tokens_raw:
                    if isinstance(t, dict):
                        tid = t.get("token_id", t.get("id", ""))
                        if tid and len(str(tid)) > 10:
                            token_ids.append(str(tid))
                    elif isinstance(t, str) and len(t) > 10:
                        token_ids.append(t)

    return token_ids


def _matches_geopolitical(market: Dict[str, Any]) -> Optional[str]:
    """Check if a market matches any geopolitical keyword.

    Returns:
        The first matched keyword, or None.
    """
    searchable = (
        market.get("question", "")
        + " " + market.get("description", "")[:500]
    ).lower()
    for kw in GEOPOLITICAL_KEYWORDS:
        if kw in searchable:
            return kw
    return None


def collect_all_geopolitical_markets() -> pd.DataFrame:
    """Scan Gamma API markets via pagination and filter for geopolitical content."""
    all_markets: Dict[str, Dict] = {}

    for page in range(MAX_PAGES):
        offset = page * PAGE_SIZE
        try:
            resp = requests.get(
                f"{GAMMA_API_BASE}/markets",
                params={"limit": PAGE_SIZE, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            markets = resp.json()
        except requests.RequestException as e:
            logger.warning("Gamma API page %d failed: %s", page, e)
            break

        if not markets:
            logger.info("No more markets at offset %d. Stopping.", offset)
            break

        for m in markets:
            mid = str(m.get("id") or m.get("condition_id", ""))
            if not mid or mid in all_markets:
                continue

            matched_kw = _matches_geopolitical(m)
            if not matched_kw:
                continue

            clob_ids = _extract_clob_token_ids(m)
            all_markets[mid] = {
                "market_id": mid,
                "question": m.get("question", ""),
                "description": m.get("description", "")[:500],
                "slug": m.get("slug", ""),
                "active": m.get("active", False),
                "closed": m.get("closed", False),
                "created_at": m.get("created_at", ""),
                "end_date": m.get("end_date_iso", ""),
                "volume": m.get("volume", 0),
                "liquidity": m.get("liquidity", 0),
                "clob_token_ids": json.dumps(clob_ids),
                "condition_id": m.get("conditionId", m.get("condition_id", "")),
                "keyword_match": matched_kw,
            }

        if page % 10 == 0:
            logger.info(
                "Page %d (offset=%d): scanned %d markets, %d geo matches so far.",
                page, offset, (page + 1) * PAGE_SIZE, len(all_markets),
            )
        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(list(all_markets.values()))
    n_with_tokens = 0
    if not df.empty:
        n_with_tokens = df["clob_token_ids"].apply(
            lambda x: len(json.loads(x)) > 0 if x else False
        ).sum()
    logger.info(
        "Found %d unique geopolitical markets (%d with CLOB token IDs).",
        len(df), n_with_tokens,
    )
    return df


# ---------- CLOB API: Historical Prices ----------

def get_price_history(
    token_id: str,
    fidelity: int = 1440,  # daily candles
) -> List[Dict[str, Any]]:
    """Fetch historical price candles for a token from CLOB API."""
    url = f"{CLOB_API_BASE}/prices-history"
    params = {"market": token_id, "interval": "max", "fidelity": fidelity}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "history" in data:
            return data["history"]
        if isinstance(data, list):
            return data
        return []
    except requests.RequestException as e:
        logger.warning("CLOB price history failed for token %s: %s", token_id[:20], e)
        return []


def collect_price_histories(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Collect price histories for all discovered markets."""
    all_rows: List[Dict] = []
    markets_with_data = 0
    skipped = 0

    for idx, row in markets_df.iterrows():
        market_id = row["market_id"]
        raw = row.get("clob_token_ids", "[]")

        # Parse CLOB token IDs (handle potential double-encoding)
        parsed = raw
        for _ in range(3):
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except (json.JSONDecodeError, TypeError):
                    break
            else:
                break

        token_ids: List[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    tid = item.get("token_id", item.get("id", ""))
                    if tid and len(str(tid)) > 10:
                        token_ids.append(str(tid))
                elif isinstance(item, str) and len(item) > 10:
                    token_ids.append(item)

        if not token_ids:
            skipped += 1
            continue

        market_has_data = False
        for i, tid in enumerate(token_ids[:2]):  # Yes/No only
            outcome_label = "Yes" if i == 0 else "No"
            history = get_price_history(tid)

            if history:
                market_has_data = True
                for candle in history:
                    all_rows.append({
                        "market_id": market_id,
                        "token_id": tid,
                        "outcome": outcome_label,
                        "question": row["question"],
                        "timestamp": candle.get("t", ""),
                        "price": candle.get("p", None),
                    })

            time.sleep(RATE_LIMIT_DELAY)

        if market_has_data:
            markets_with_data += 1

        if (idx + 1) % 25 == 0:
            logger.info(
                "Progress: %d/%d markets processed, %d with data, %d observations.",
                idx + 1, len(markets_df), markets_with_data, len(all_rows),
            )

    df = pd.DataFrame(all_rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    logger.info(
        "Collected %d price observations from %d markets (%d skipped, no tokens).",
        len(df), markets_with_data, skipped,
    )
    return df


# ---------- Main ----------

def main() -> None:
    """Run full Polymarket data collection pipeline."""
    logger.info("=" * 60)
    logger.info("Polymarket Geopolitical Data Collection")
    logger.info("=" * 60)

    # Step 1: Discover markets
    markets_df = collect_all_geopolitical_markets()
    markets_path = DATA_DIR / "polymarket_markets.csv"
    markets_df.to_csv(markets_path, index=False)
    logger.info("Saved market metadata to %s", markets_path)

    if markets_df.empty:
        logger.warning("No markets found. Exiting.")
        return

    # Step 2: Collect price histories
    prices_df = collect_price_histories(markets_df)
    prices_path = DATA_DIR / "polymarket_prices.csv"
    prices_df.to_csv(prices_path, index=False)
    logger.info("Saved price histories to %s", prices_path)

    # Step 3: Summary statistics
    if not prices_df.empty:
        summary = prices_df.groupby("market_id").agg(
            question=("question", "first"),
            n_obs=("price", "count"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            first_date=("timestamp", "min"),
            last_date=("timestamp", "max"),
        ).reset_index()
        summary_path = DATA_DIR / "polymarket_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved summary statistics to %s", summary_path)

    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
