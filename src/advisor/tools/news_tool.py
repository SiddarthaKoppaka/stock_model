"""
Tool 3 — Real-Time News & Sentiment

Fetches recent financial news and computes simple keyword-based
sentiment for specified Indian equity symbols.

Sources:
    - yfinance .news property
    - Economic Times RSS (if feedparser available)
    - MoneyControl RSS (if feedparser available)
"""

import time
import datetime
import re
from typing import Dict, List, Optional
from loguru import logger

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.info("feedparser not installed — RSS sources unavailable. Install with: pip install feedparser")


# ── Keyword-based sentiment ─────────────────────────────────────────────────

_POSITIVE_KEYWORDS = {
    'upgrade', 'bullish', 'beats', 'record', 'profit', 'growth', 'surge',
    'rally', 'outperform', 'strong', 'gain', 'optimistic', 'boost',
    'dividend', 'buyback', 'expansion', 'recovery', 'breakthrough',
    'approval', 'award', 'partnership', 'deal', 'innovation',
}

_NEGATIVE_KEYWORDS = {
    'downgrade', 'bearish', 'misses', 'loss', 'decline', 'crash', 'slump',
    'underperform', 'weak', 'fall', 'pessimistic', 'sell-off', 'selloff',
    'fraud', 'investigation', 'penalty', 'fine', 'default', 'bankruptcy',
    'recall', 'litigation', 'layoff', 'shutdown', 'warning',
}

_HIGH_IMPACT_CATEGORIES = {
    'earnings', 'acquisition', 'regulatory', 'promoter', 'insider',
    'merger', 'takeover', 'sebi', 'rbi', 'government', 'policy',
}


def _classify_sentiment(headline: str) -> str:
    """Classify headline sentiment as positive/negative/neutral."""
    words = set(headline.lower().split())
    pos_hits = words & _POSITIVE_KEYWORDS
    neg_hits = words & _NEGATIVE_KEYWORDS

    if len(pos_hits) > len(neg_hits):
        return 'positive'
    elif len(neg_hits) > len(pos_hits):
        return 'negative'
    return 'neutral'


def _get_impact_categories(headline: str) -> List[str]:
    """Flag high-impact categories present in headline."""
    text_lower = headline.lower()
    return [cat for cat in _HIGH_IMPACT_CATEGORIES if cat in text_lower]


def _fetch_yfinance_news(symbol: str, lookback_hours: int) -> List[Dict]:
    """Fetch news from yfinance for a symbol."""
    if not YF_AVAILABLE:
        return []

    try:
        yf_sym = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
        ticker = yf.Ticker(yf_sym)
        news_items = ticker.news or []

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=lookback_hours)
        results = []

        for item in news_items[:10]:  # check up to 10
            # yfinance news format varies; handle both dict styles
            title = item.get('title', item.get('headline', ''))
            pub_time = item.get('providerPublishTime', item.get('publishedDate', 0))

            if isinstance(pub_time, (int, float)):
                pub_dt = datetime.datetime.fromtimestamp(pub_time, tz=datetime.timezone.utc)
            elif isinstance(pub_time, str):
                try:
                    pub_dt = datetime.datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                except ValueError:
                    pub_dt = datetime.datetime.now(datetime.timezone.utc)
            else:
                pub_dt = datetime.datetime.now(datetime.timezone.utc)

            if pub_dt >= cutoff and title:
                results.append({
                    'headline': title,
                    'source': item.get('publisher', item.get('source', 'yfinance')),
                    'timestamp': pub_dt.isoformat(),
                    'sentiment': _classify_sentiment(title),
                    'impact_categories': _get_impact_categories(title),
                })

            if len(results) >= 5:
                break

        return results

    except Exception as e:
        logger.debug(f"yfinance news failed for {symbol}: {e}")
        return []


def _fetch_rss_news(symbol: str, lookback_hours: int) -> List[Dict]:
    """Fetch news from Economic Times and MoneyControl RSS feeds."""
    if not FEEDPARSER_AVAILABLE:
        return []

    results = []
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=lookback_hours)

    rss_feeds = [
        (f"https://economictimes.indiatimes.com/topic/{symbol}/rss", "Economic Times"),
        (f"https://www.moneycontrol.com/rss/companynews.xml", "MoneyControl"),
    ]

    for url, source_name in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in (feed.entries or [])[:10]:
                title = entry.get('title', '')
                # Check if headline mentions the symbol
                if symbol.upper() not in title.upper() and source_name == "MoneyControl":
                    continue

                pub_time = entry.get('published_parsed')
                if pub_time:
                    import calendar
                    pub_dt = datetime.datetime.fromtimestamp(
                        calendar.timegm(pub_time), tz=datetime.timezone.utc
                    )
                else:
                    pub_dt = datetime.datetime.now(datetime.timezone.utc)

                if pub_dt >= cutoff and title:
                    results.append({
                        'headline': title,
                        'source': source_name,
                        'timestamp': pub_dt.isoformat(),
                        'sentiment': _classify_sentiment(title),
                        'impact_categories': _get_impact_categories(title),
                    })

                if len(results) >= 3:
                    break
        except Exception as e:
            logger.debug(f"RSS feed failed for {symbol} ({source_name}): {e}")

    return results


def get_stock_news(
    symbols: List[str],
    lookback_hours: int = 48,
) -> Dict:
    """
    Fetch and analyse recent financial news for specified stocks.

    Args:
        symbols: list of NSE stock symbols
        lookback_hours: how far back to look for news (default 48h)

    Returns:
        Dict with per-symbol news, sentiment summary, and metadata
    """
    logger.info(f"Fetching news for {len(symbols)} symbols (lookback {lookback_hours}h) …")

    all_news: Dict[str, List[Dict]] = {}
    sentiment_summary: Dict[str, str] = {}

    for sym in symbols:
        clean_sym = sym.replace('.NS', '')

        try:
            # Combine sources
            news_items: List[Dict] = []
            news_items.extend(_fetch_yfinance_news(clean_sym, lookback_hours))
            news_items.extend(_fetch_rss_news(clean_sym, lookback_hours))

            # Deduplicate by headline similarity
            seen_titles = set()
            unique_news = []
            for item in news_items:
                title_key = re.sub(r'\W+', '', item['headline'].lower())[:50]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_news.append(item)

            # Keep top 5
            unique_news = unique_news[:5]
            all_news[clean_sym] = unique_news

            # Overall sentiment for symbol
            if unique_news:
                sentiments = [n['sentiment'] for n in unique_news]
                pos = sentiments.count('positive')
                neg = sentiments.count('negative')
                if pos > neg:
                    sentiment_summary[clean_sym] = 'positive'
                elif neg > pos:
                    sentiment_summary[clean_sym] = 'negative'
                else:
                    sentiment_summary[clean_sym] = 'neutral'
            else:
                sentiment_summary[clean_sym] = 'no_data'

        except Exception as e:
            logger.warning(f"News fetch failed for {clean_sym}: {e}")
            all_news[clean_sym] = []
            sentiment_summary[clean_sym] = 'error'

    return {
        'as_of': datetime.datetime.now().isoformat(),
        'lookback_hours': lookback_hours,
        'news': all_news,
        'sentiment_summary': sentiment_summary,
        'symbols_with_news': sum(1 for v in all_news.values() if v),
        'symbols_no_news': sum(1 for v in all_news.values() if not v),
    }


# ── Tool schema for Claude ──────────────────────────────────────────────────

NEWS_TOOL_SCHEMA = {
    "name": "get_stock_news",
    "description": (
        "Fetch recent financial news headlines and sentiment for specified "
        "Indian stock symbols. Returns up to 5 headlines per symbol with "
        "source, timestamp, sentiment (positive/negative/neutral), and "
        "high-impact category flags (earnings, acquisition, regulatory, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of NSE stock symbols (e.g. ['RELIANCE', 'TCS'])",
            },
            "lookback_hours": {
                "type": "integer",
                "description": "How many hours back to search for news (default 48)",
                "default": 48,
            },
        },
        "required": ["symbols"],
    },
}


if __name__ == '__main__':
    print("News Tool — Standalone Test")
    print("=" * 60)
    result = get_stock_news(['RELIANCE', 'TCS', 'INFY'], lookback_hours=48)
    print(f"With news: {result['symbols_with_news']}, No news: {result['symbols_no_news']}")
    for sym, items in result['news'].items():
        print(f"\n  {sym} — overall: {result['sentiment_summary'][sym]}")
        for item in items:
            cats = ', '.join(item['impact_categories']) if item['impact_categories'] else '—'
            print(f"    [{item['sentiment']}] {item['headline'][:80]}…  (cats: {cats})")
