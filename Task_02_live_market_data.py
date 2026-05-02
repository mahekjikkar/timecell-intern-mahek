"""
Task_02_live_market_data.py
===================
Live Market Data Fetcher

Fetches real-time prices for:
  - NIFTY 50          (Index,   via yfinance)
  - Reliance Industries (Equity, via yfinance)
  - Bitcoin           (Crypto,  via CoinGecko public API)

Dependencies
------------
    pip install yfinance requests rich

"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
import yfinance as yf
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor


# Logging — use Rich's handler so log output is consistent with the UI layer

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log: logging.Logger = logging.getLogger("market_data")


# Constants
COINGECKO_API_URL: str = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin&vs_currencies=usd"
)
REQUEST_TIMEOUT_SECONDS: int = 10

# Colour thresholds used by the presentation layer (purely cosmetic)
_POSITIVE_STYLE = "bold green"
_NEUTRAL_STYLE = "bold yellow"
_ERROR_STYLE = "bold red"


# Domain model
@dataclass(frozen=True)
class AssetQuote:
   
    name: str
    symbol: str
    price: Optional[float]
    currency: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str = ""

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when a price was successfully fetched."""
        return self.price is not None



# Data-fetching layer
def _utcnow() -> datetime:
    """Return the current UTC-aware datetime (thin wrapper for testability)."""
    return datetime.now(timezone.utc)


def fetch_yfinance_quote(
    name: str,
    ticker: str,
    currency: str,
) -> AssetQuote:
    
    """Fetch the current price for a given ticker using yfinance."""
    fetched_at = _utcnow()
    try:
        ticker_obj = yf.Ticker(ticker)
        info: dict = ticker_obj.fast_info  # type: ignore[assignment]

        # fast_info exposes `last_price`; fall back to previous close
        price: Optional[float] = (
            info.get("last_price")
            or info.get("previous_close")
        )

        if price is None:
            # Last resort: pull the most recent 1-day history
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if price is None:
            raise ValueError(
                f"No price data returned by yfinance for ticker '{ticker}'."
            )

        return AssetQuote(
            name=name,
            symbol=ticker,
            price=round(float(price), 2),
            currency=currency,
            fetched_at=fetched_at,
        )

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[yellow]⚠ Could not fetch [bold]%s[/bold] (%s): %s[/yellow]",
            name,
            ticker,
            exc,
        )
        return AssetQuote(
            name=name,
            symbol=ticker,
            price=None,
            currency=currency,
            fetched_at=fetched_at,
            error=str(exc),
        )


def fetch_bitcoin_quote() -> AssetQuote:
    
    fetched_at = _utcnow()
    try:
        response = requests.get(COINGECKO_API_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        payload: dict = response.json()
        price_usd: Optional[float] = (
            payload.get("bitcoin", {}).get("usd")
        )

        if price_usd is None:
            raise ValueError(
                "Unexpected CoinGecko response schema — 'bitcoin.usd' key missing."
            )

        return AssetQuote(
            name="Bitcoin",
            symbol="BTC-USD",
            price=round(float(price_usd), 2),
            currency="USD",
            fetched_at=fetched_at,
        )

    except requests.exceptions.Timeout:
        msg = f"CoinGecko request timed out after {REQUEST_TIMEOUT_SECONDS}s."
        log.warning("[yellow]⚠ Could not fetch [bold]Bitcoin[/bold]: %s[/yellow]", msg)
        return AssetQuote(
            name="Bitcoin",
            symbol="BTC-USD",
            price=None,
            currency="USD",
            fetched_at=fetched_at,
            error=msg,
        )
    except requests.exceptions.HTTPError as exc:
        msg = f"HTTP {exc.response.status_code} from CoinGecko."
        log.warning("[yellow]⚠ Could not fetch [bold]Bitcoin[/bold]: %s[/yellow]", msg)
        return AssetQuote(
            name="Bitcoin",
            symbol="BTC-USD",
            price=None,
            currency="USD",
            fetched_at=fetched_at,
            error=msg,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[yellow]⚠ Could not fetch [bold]Bitcoin[/bold]: %s[/yellow]", exc
        )
        return AssetQuote(
            name="Bitcoin",
            symbol="BTC-USD",
            price=None,
            currency="USD",
            fetched_at=fetched_at,
            error=str(exc),
        )

def fetch_all_quotes() -> list[AssetQuote]:
    """Fetch all assets concurrently. Independent I/O, safe to parallelize."""
    tasks = [
        (fetch_yfinance_quote, ("NIFTY 50", "^NSEI", "INR")),
        (fetch_yfinance_quote, ("Reliance Industries", "RELIANCE.NS", "INR")),
        (fetch_bitcoin_quote, ()),
    ]
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = [ex.submit(fn, *args) for fn, args in tasks]
        return [f.result() for f in futures]


# Presentation layer
_CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
}


def _format_price(quote: AssetQuote) -> Text:
    
    if not quote.is_valid:
        return Text("N/A", style=_ERROR_STYLE)

    symbol = _CURRENCY_SYMBOLS.get(quote.currency, "")
    formatted = f"{symbol}{quote.price:,.2f}"
    return Text(formatted, style=_POSITIVE_STYLE)


def _format_timestamp(quote: AssetQuote) -> str:
    return quote.fetched_at.astimezone(ZoneInfo("Asia/Kolkata")).strftime(
        "%Y-%m-%d %H:%M:%S IST"
    )


def _build_table(quotes: list[AssetQuote]) -> Table:

    table = Table(
        title="📈  Live Market Snapshot",
        box=box.DOUBLE_EDGE,
        border_style="bright_blue",
        header_style="bold bright_white on dark_blue",
        title_style="bold cyan",
        show_lines=True,
        expand=False,
    )

    table.add_column("Asset", style="bold white", min_width=22, no_wrap=True)
    table.add_column("Symbol", style="dim white", min_width=14, no_wrap=True)
    table.add_column("Current Price", justify="right", min_width=16)
    table.add_column("Currency", justify="center", min_width=10)
    table.add_column("Fetched At (IST)", justify="center", min_width=22)
    table.add_column("Status", justify="center", min_width=10)

    for quote in quotes:
        status_text = (
            Text("✔  OK", style=_POSITIVE_STYLE)
            if quote.is_valid
            else Text("✘  ERR", style=_ERROR_STYLE)
        )
        table.add_row(
            quote.name,
            quote.symbol,
            _format_price(quote),
            quote.currency,
            _format_timestamp(quote),
            status_text,
        )

    return table


def render_market_dashboard(quotes: list[AssetQuote]) -> None:

    console = Console()
    console.print()

    # ── Header ──────────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            "[bold cyan]Wealth Intelligence Platform[/bold cyan]  "
            "[dim]|  Live Market Data Feed[/dim]",
            border_style="bright_blue",
            padding=(0, 2),
        )
    )
    console.print()

    # ── Main table ──────────────────────────────────────────────────────────
    console.print(_build_table(quotes))

    # ── Error detail block (only shown when at least one fetch failed) ──────
    failed = [q for q in quotes if not q.is_valid]
    if failed:
        console.print()
        console.print(
            Panel(
                "\n".join(
                    f"[bold red]{q.name}[/bold red]: {q.error}" for q in failed
                ),
                title="[bold red]⚠  Fetch Errors[/bold red]",
                border_style="red",
                padding=(0, 2),
            )
        )

    # ── Footer ───────────────────────────────────────────────────────────────
    console.print()
    success_count = sum(1 for q in quotes if q.is_valid)
    console.print(
        f"  [dim]{success_count}/{len(quotes)} assets fetched successfully  "
        f"·  Data sourced from Yahoo Finance & CoinGecko  "
        f"·  Prices are indicative[/dim]"
    )
    console.print()


def main() -> None:

    console = Console()
    console.print(
        "\n[dim]Fetching live market data — please wait…[/dim]", end="\n\n"
    )

    quotes = fetch_all_quotes()
    render_market_dashboard(quotes)

    # Exit with a non-zero code if every single asset failed (aids CI pipelines)
    if not any(q.is_valid for q in quotes):
        log.error("All asset fetches failed. Check network connectivity.")
        sys.exit(1)


if __name__ == "__main__":
    main()
