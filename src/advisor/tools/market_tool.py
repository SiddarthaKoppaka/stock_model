"""
Tool 2 — Real-Time Market Data

Fetches live NSE data using yfinance with caching, retry logic,
and graceful per-symbol failure handling.
"""

import time
import datetime
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from loguru import logger


# ── Simple TTL cache ─────────────────────────────────────────────────────────

_market_cache: Dict[str, Dict] = {}
_cache_timestamp: float = 0.0
_CACHE_TTL_SECONDS = 900  # 15 minutes


def _is_cache_valid() -> bool:
    return (time.time() - _cache_timestamp) < _CACHE_TTL_SECONDS


def _clear_cache():
    global _market_cache, _cache_timestamp
    _market_cache = {}
    _cache_timestamp = 0.0


def _fetch_single_symbol(symbol: str, lookback_days: int, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch market data for a single NSE symbol with retry logic.

    Args:
        symbol: NSE stock symbol (with or without .NS suffix)
        lookback_days: number of days to fetch
        max_retries: retry attempts on failure

    Returns:
        dict with price/volume data, or None on failure
    """
    # Ensure .NS suffix for yfinance
    yf_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"

    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=f"{max(lookback_days + 10, 30)}d")

            if hist.empty or len(hist) < 2:
                logger.warning(f"{yf_symbol}: No data returned")
                return None

            hist = hist.tail(lookback_days + 1)

            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            volume = int(hist['Volume'].iloc[-1])

            # 5-day return
            if len(hist) >= 6:
                ret_5d = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-6]) - 1)
            else:
                ret_5d = float((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1)

            # 52-week high/low positioning
            try:
                hist_52w = ticker.history(period="1y")
                if not hist_52w.empty:
                    high_52w = float(hist_52w['High'].max())
                    low_52w = float(hist_52w['Low'].min())
                    pct_from_high = (current_price - high_52w) / high_52w if high_52w > 0 else 0
                    pct_from_low = (current_price - low_52w) / low_52w if low_52w > 0 else 0
                else:
                    high_52w = low_52w = pct_from_high = pct_from_low = None
            except Exception:
                high_52w = low_52w = pct_from_high = pct_from_low = None

            # Average daily volume (20-day)
            adv_20d = float(hist['Volume'].tail(20).mean()) if len(hist) >= 20 else float(hist['Volume'].mean())

            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'prev_close': round(prev_close, 2),
                'daily_change_pct': round(100 * (current_price - prev_close) / prev_close, 2),
                'volume': volume,
                'return_5d': round(ret_5d, 4),
                'high_52w': round(high_52w, 2) if high_52w else None,
                'low_52w': round(low_52w, 2) if low_52w else None,
                'pct_from_52w_high': round(pct_from_high, 4) if pct_from_high is not None else None,
                'pct_from_52w_low': round(pct_from_low, 4) if pct_from_low is not None else None,
                'adv_20d': round(adv_20d, 0),
            }

        except Exception as e:
            logger.warning(f"{yf_symbol}: Attempt {attempt + 1} failed — {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def _fetch_nifty500_index() -> Optional[Dict]:
    """Fetch NIFTY500 index level and return for market context."""
    try:
        ticker = yf.Ticker("^CRSLDX")  # NIFTY 500 index
        hist = ticker.history(period="10d")

        if hist.empty:
            # Fallback to NIFTY 50
            ticker = yf.Ticker("^NSEI")
            hist = ticker.history(period="10d")

        if hist.empty:
            return None

        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current

        return {
            'index': 'NIFTY500',
            'current_level': round(current, 2),
            'daily_change_pct': round(100 * (current - prev) / prev, 2),
            'return_5d': round(
                float((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1), 4
            ) if len(hist) >= 5 else None,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch NIFTY500 index: {e}")
        return None


def get_market_data(
    symbols: List[str],
    lookback_days: int = 5,
) -> Dict:
    """
    Fetch real-time NSE market data for a list of symbols.

    Args:
        symbols: list of NSE stock symbols (with or without .NS suffix)
        lookback_days: number of trading days to look back

    Returns:
        Dict with per-symbol data and market context
    """
    global _market_cache, _cache_timestamp

    # Check cache
    cache_key = f"{','.join(sorted(symbols))}_{lookback_days}"
    if _is_cache_valid() and cache_key in _market_cache:
        logger.info("Returning cached market data")
        return _market_cache[cache_key]

    logger.info(f"Fetching market data for {len(symbols)} symbols …")

    stock_data = {}
    failed_symbols = []

    for sym in symbols:
        clean_sym = sym.replace('.NS', '')
        data = _fetch_single_symbol(clean_sym, lookback_days)
        if data is not None:
            stock_data[clean_sym] = data
        else:
            failed_symbols.append(clean_sym)
            stock_data[clean_sym] = None

    market_context = _fetch_nifty500_index()

    result = {
        'as_of': datetime.datetime.now().isoformat(),
        'stocks': stock_data,
        'market_context': market_context,
        'failed_symbols': failed_symbols,
        'symbols_fetched': len(stock_data) - len(failed_symbols),
        'symbols_failed': len(failed_symbols),
    }

    # Update cache
    _market_cache[cache_key] = result
    _cache_timestamp = time.time()

    return result


# ── Tool schema for Claude ──────────────────────────────────────────────────

MARKET_TOOL_SCHEMA = {
    "name": "get_market_data",
    "description": (
        "Fetch real-time NSE market data for specified stock symbols. "
        "Returns current price, volume, 5-day return, 52-week high/low position, "
        "and average daily volume. Also provides NIFTY500 index context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of NSE stock symbols (e.g. ['RELIANCE', 'TCS', 'INFY'])",
            },
            "lookback_days": {
                "type": "integer",
                "description": "Number of trading days to look back (default 5)",
                "default": 5,
            },
        },
        "required": ["symbols"],
    },
}


if __name__ == '__main__':
    print("Market Tool — Standalone Test")
    print("=" * 60)
    result = get_market_data(['RELIANCE', 'TCS', 'INFY'], lookback_days=5)
    print(f"Fetched: {result['symbols_fetched']}, Failed: {result['symbols_failed']}")
    for sym, data in result['stocks'].items():
        if data:
            print(f"  {sym}: ₹{data['current_price']}  5d: {data['return_5d']:.2%}")
        else:
            print(f"  {sym}: FAILED")
    if result['market_context']:
        print(f"\n  Market: {result['market_context']['index']} = {result['market_context']['current_level']}")
