"""
Real Return Auditor
===================

A terminal tool that strips four layers of return erosion — tax, inflation,
and currency — to show what an Indian family actually keeps.

The premise: every retail tracker shows nominal returns. Almost no tool
shows what survives the four layers of erosion that matter to an HNI
family — Indian tax, Indian inflation (CPI), and INR depreciation against
the USD. This tool runs the four-layer waterfall on a small portfolio and
prints the result as a calm, CIO-style report.

Three deliberate design choices, called out so a reviewer can challenge them:

    1. Asset returns are hardcoded long-run class averages, not user-entered
       actual returns. The tool teaches structural truths about asset
       classes, not personal performance. Easier to compare apples-to-apples.

    2. We assume positions are realised today: tax is applied in full on
       accumulated gains. Unrealised gains are not real wealth until they
       can be harvested, and a "tax-deferred" framing flatters every asset.

    3. Inflation and FX adjustments use the Fisher relation, not subtraction:
           real = (1 + nominal) / (1 + drag) − 1
       Subtraction is a fine mental shortcut but breaks for large numbers.

Requirements
------------
    Python 3.10+ (uses ``match`` and PEP 604 union syntax)
    rich           (``pip install rich``)

Usage
-----
    python real_return_auditor.py --demo
    python real_return_auditor.py --portfolio sample_portfolio.json
    python real_return_auditor.py --portfolio sample_portfolio.json --cpi 0.065 --fx 0.04

"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Final

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ════════════════════════════════════════════════════════════════════
# CONFIG
# All assumptions live here. A reviewer should be able to read this
# block alone and understand every number the tool prints.
# ════════════════════════════════════════════════════════════════════

class AssetClass(str, Enum):
    """Asset classes recognised by the auditor.

    String values double as the CSV input vocabulary so users can
    type ``equity_in`` directly in their portfolio file.
    """

    EQUITY_IN = "equity_in"
    EQUITY_US = "equity_us"
    DEBT_MF = "debt_mf"
    FD = "fd"
    GOLD = "gold"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"


# Long-run nominal CAGRs in INR terms. Indicative, not promises.
# Sources: Nifty 50 TRI (NSE 20-yr avg), S&P 500 (S&P + INR depreciation),
# RBI weighted FD rates, MCX gold, RBI house price index, BTC INR public data.
NOMINAL_CAGR: Final[dict[AssetClass, float]] = {
    AssetClass.EQUITY_IN:   0.12,
    AssetClass.EQUITY_US:   0.13,   # ~10% USD return + ~3% INR depreciation
    AssetClass.DEBT_MF:     0.07,
    AssetClass.FD:          0.07,
    AssetClass.GOLD:        0.09,
    AssetClass.REAL_ESTATE: 0.07,
    AssetClass.CRYPTO:      0.40,   # heavy caveat; long-run BTC INR
}

ASSET_CLASS_LABELS: Final[dict[AssetClass, str]] = {
    AssetClass.EQUITY_IN:   "Indian Equity",
    AssetClass.EQUITY_US:   "US Equity (LRS)",
    AssetClass.DEBT_MF:     "Debt MF",
    AssetClass.FD:          "Fixed Deposit",
    AssetClass.GOLD:        "Gold",
    AssetClass.REAL_ESTATE: "Real Estate",
    AssetClass.CRYPTO:      "Crypto",
}

# Macro defaults — overridable from CLI.
INDIA_CPI: Final[float] = 0.06              # RBI long-run CPI band midpoint
INR_USD_DEPRECIATION: Final[float] = 0.035  # 20-yr INR/USD trend
HNI_SLAB_RATE: Final[float] = 0.30          # top slab; surcharge ignored


def effective_tax_rate(asset_class: AssetClass, years_held: float) -> float:
    """Return the marginal tax rate applied to gains at realisation.

    Reflects the post-Budget-2024 Indian tax regime:

    - Indian equity ``> 1y``      : 12.5% LTCG (₹1.25L exemption ignored —
                                     typically already consumed in HNI books).
    - Indian equity ``≤ 1y``      : 20% STCG.
    - Foreign equity ``> 2y``     : 12.5% LTCG.   ``≤ 2y``: slab.
    - Debt MF / FD               : always slab (post-April-2023 regime).
    - Gold / Real estate ``> 2y`` : 12.5% LTCG, no indexation.
    - Gold / Real estate ``≤ 2y`` : slab.
    - Crypto (VDA)               : flat 30%, regardless of horizon.
    """
    match asset_class:
        case AssetClass.EQUITY_IN:
            return 0.125 if years_held > 1 else 0.20
        case AssetClass.EQUITY_US:
            return 0.125 if years_held > 2 else HNI_SLAB_RATE
        case AssetClass.DEBT_MF | AssetClass.FD:
            return HNI_SLAB_RATE
        case AssetClass.GOLD | AssetClass.REAL_ESTATE:
            return 0.125 if years_held > 2 else HNI_SLAB_RATE
        case AssetClass.CRYPTO:
            return 0.30


# ════════════════════════════════════════════════════════════════════
# DATA TYPES
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Asset:
    """A single holding in the portfolio."""

    name: str
    asset_class: AssetClass
    capital: float       # INR invested at year 0
    years_held: float    # holding period; drives tax bucket & compounding


@dataclass(frozen=True)
class WaterfallResult:
    """The four-layer return waterfall for a single asset.

    All CAGRs are decimal (0.07 means 7%). All terminal values are in
    rupees. The ``real_*`` terminals are expressed in *today's*
    purchasing power, not nominal year-T rupees.
    """

    asset: Asset
    nominal_cagr: float
    post_tax_cagr: float
    real_inr_cagr: float
    real_usd_cagr: float
    nominal_terminal: float
    post_tax_terminal: float
    real_inr_terminal: float
    real_usd_terminal: float
    tax_rate: float
    breakeven_cpi: float    # CPI at which real INR return = 0


# ════════════════════════════════════════════════════════════════════
# MATH PIPELINE
# Pure functions. No I/O. Easy to unit-test, easy to defend in a review.
# ════════════════════════════════════════════════════════════════════

def compute_waterfall(
    asset: Asset,
    nominal_cagr: float,
    cpi: float = INDIA_CPI,
    fx_drag: float = INR_USD_DEPRECIATION,
) -> WaterfallResult:
    """Run the four-layer waterfall on a single asset.

    Layers, in order:

    1. **Nominal**   — headline compounded return.
    2. **Post-tax**  — marginal rate applied to gains, capital untaxed.
    3. **Real INR**  — Fisher-deflated by India CPI.
    4. **Real USD**  — Fisher-deflated by INR/USD trend.

    Raises:
        ValueError: if capital or holding period are non-positive.
    """
    if asset.years_held <= 0:
        raise ValueError(f"{asset.name!r}: years_held must be > 0.")
    if asset.capital <= 0:
        raise ValueError(f"{asset.name!r}: capital must be > 0.")

    c, t, r_n = asset.capital, asset.years_held, nominal_cagr

    # Layer 1 — nominal terminal value.
    v_nominal = c * (1 + r_n) ** t
    gains = v_nominal - c

    # Layer 2 — post-tax. Tax falls on gains, not capital.
    tax = effective_tax_rate(asset.asset_class, t)
    v_post_tax = c + gains * (1 - tax)
    r_post_tax = (v_post_tax / c) ** (1 / t) - 1

    # Layer 3 — real INR (Fisher).
    r_real_inr = (1 + r_post_tax) / (1 + cpi) - 1
    v_real_inr = c * (1 + r_real_inr) ** t

    # Layer 4 — real USD (Fisher again, against INR depreciation).
    r_real_usd = (1 + r_real_inr) / (1 + fx_drag) - 1
    v_real_usd = c * (1 + r_real_usd) ** t

    # Inverted-thinking: at what CPI does real INR return = 0?
    # Solving (1 + r_post_tax) / (1 + cpi*) − 1 = 0 gives cpi* = r_post_tax.
    breakeven_cpi = r_post_tax

    return WaterfallResult(
        asset=asset,
        nominal_cagr=r_n,
        post_tax_cagr=r_post_tax,
        real_inr_cagr=r_real_inr,
        real_usd_cagr=r_real_usd,
        nominal_terminal=v_nominal,
        post_tax_terminal=v_post_tax,
        real_inr_terminal=v_real_inr,
        real_usd_terminal=v_real_usd,
        tax_rate=tax,
        breakeven_cpi=breakeven_cpi,
    )


def aggregate_portfolio(results: list[WaterfallResult]) -> dict[str, float]:
    """Roll per-asset waterfalls up to portfolio-level metrics.

    Two summary statistics are produced for each layer:

    - **CAGR**: capital-weighted average of asset CAGRs. Reading: "the
      average rate at which a rupee in this portfolio earned per year".
      This avoids the ambiguity of computing one CAGR over a mixed-horizon
      portfolio, which has no clean closed form.

    - **Terminal value**: simple sum of asset terminals at their own
      holding periods. Reading: "what this portfolio is worth today,
      under each filter".

    Returns an empty dict for an empty portfolio.
    """
    if not results:
        return {}

    total_capital = sum(r.asset.capital for r in results)
    weighted_years = (
        sum(r.asset.capital * r.asset.years_held for r in results) / total_capital
    )

    def w_avg(attr: str) -> float:
        return sum(r.asset.capital * getattr(r, attr) for r in results) / total_capital

    return {
        "capital":           total_capital,
        "horizon":           weighted_years,
        "nominal_terminal":  sum(r.nominal_terminal for r in results),
        "post_tax_terminal": sum(r.post_tax_terminal for r in results),
        "real_inr_terminal": sum(r.real_inr_terminal for r in results),
        "real_usd_terminal": sum(r.real_usd_terminal for r in results),
        "nominal_cagr":      w_avg("nominal_cagr"),
        "post_tax_cagr":     w_avg("post_tax_cagr"),
        "real_inr_cagr":     w_avg("real_inr_cagr"),
        "real_usd_cagr":     w_avg("real_usd_cagr"),
    }


# ════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ════════════════════════════════════════════════════════════════════

def format_inr(amount: float) -> str:
    """Format rupees in lakh / crore notation, with sign."""
    if amount < 0:
        return f"-{format_inr(-amount)}"
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.2f} L"
    return f"₹{amount:,.0f}"


def format_pct_signed(rate: float, decimals: int = 1) -> str:
    """Format a decimal rate as a signed percentage (e.g. ``+9.4%``, ``-2.4%``)."""
    return f"{rate * 100:+.{decimals}f}%"


def format_pct(rate: float, decimals: int = 1) -> str:
    """Format a decimal rate as an unsigned percentage (e.g. ``9.4%``)."""
    return f"{rate * 100:.{decimals}f}%"


def cagr_color(cagr: float) -> str:
    """Map a CAGR to a rich color name. Green = healthy, yellow = thin, red = negative."""
    if cagr < 0:
        return "red"
    if cagr < 0.04:
        return "yellow"
    return "green"


# ════════════════════════════════════════════════════════════════════
# DATA INPUT
# ════════════════════════════════════════════════════════════════════

def load_portfolio_csv(path: Path) -> list[Asset]:
    """Load a portfolio from a CSV file.

    Expected columns: ``name, asset_class, capital, years_held``.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: on malformed rows or unknown asset classes.
    """
    valid_classes = ", ".join(c.value for c in AssetClass)
    assets: list[Asset] = []

    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"name", "asset_class", "capital", "years_held"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}")

            for line_no, row in enumerate(reader, start=2):
                name = (row.get("name") or "").strip()
                if not name:
                    raise ValueError(f"Line {line_no}: empty 'name'.")
                try:
                    cls = AssetClass(row["asset_class"].strip().lower())
                except ValueError:
                    raise ValueError(
                        f"Line {line_no} ({name}): unknown asset_class "
                        f"{row['asset_class']!r}. Valid: {valid_classes}"
                    ) from None
                try:
                    capital = float(row["capital"])
                    years = float(row["years_held"])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Line {line_no} ({name}): capital and years_held "
                        f"must be numeric."
                    ) from None
                assets.append(Asset(name=name, asset_class=cls,
                                    capital=capital, years_held=years))
    except FileNotFoundError:
        raise FileNotFoundError(f"Portfolio file not found: {path}") from None

    if not assets:
        raise ValueError(f"No assets loaded from {path}.")
    return assets


def sample_portfolio() -> list[Asset]:
    """A representative HNI portfolio for the ``--demo`` flag.

    Deliberately diverse: covers every asset class so the report
    showcases the full waterfall behaviour.
    """
    return [
        Asset("HDFC Bank FD",           AssetClass.FD,          50_00_000,    5),
        Asset("Nifty 50 Index Fund",    AssetClass.EQUITY_IN,   80_00_000,    5),
        Asset("Mumbai Apartment",       AssetClass.REAL_ESTATE, 1_50_00_000, 10),
        Asset("Sovereign Gold Bonds",   AssetClass.GOLD,        25_00_000,    8),
        Asset("Vanguard S&P 500 (LRS)", AssetClass.EQUITY_US,   40_00_000,    4),
        Asset("Bitcoin",                AssetClass.CRYPTO,      15_00_000,    5),
    ]


# ════════════════════════════════════════════════════════════════════
# RENDERING
# Each render function takes a Console + data, prints, returns None.
# Pure presentation. Math is done.
# ════════════════════════════════════════════════════════════════════

def render_header(console: Console, cpi: float, fx: float) -> None:
    """Title and assumption banner. Assumptions are not negotiable here —
    a wealth manager cannot use a number whose assumptions are hidden."""
    console.print(Text("§  REAL RETURN AUDITOR", style="bold"))
    console.print(Text(
        "What you actually kept after tax, inflation, and currency drag.",
        style="dim italic",
    ))
    console.print()
    today = date.today().strftime("%d %b %Y")
    console.print(
        f"[dim]Run date:[/] {today}    "
        f"[dim]CPI:[/] {format_pct(cpi)}    "
        f"[dim]INR/USD drag:[/] {format_pct(fx)}    "
        f"[dim]Tax:[/] HNI top slab"
    )
    console.print()


def render_waterfall_table(console: Console, results: list[WaterfallResult]) -> None:
    """Per-asset waterfall — the heart of the report.

    Eyes naturally jump down the Real INR / Real USD columns; that's where
    the colour coding lives, and that's where the truth is.
    """
    console.print(Text("§  PER-ASSET WATERFALL", style="bold"))

    table = Table(box=box.SIMPLE_HEAVY, padding=(0, 1), show_header=True,
                  header_style="bold")
    table.add_column("Asset", no_wrap=True)
    table.add_column("Capital", justify="right")
    table.add_column("Tax", justify="right", style="dim")
    table.add_column("Nominal", justify="right")
    table.add_column("Post-Tax", justify="right")
    table.add_column("Real INR", justify="right")
    table.add_column("Real USD", justify="right")

    for r in results:
        table.add_row(
           f"{r.asset.name} [dim]({r.asset.years_held:.0f}y)[/]",
           format_inr(r.asset.capital),
           format_pct(r.tax_rate),
           format_pct_signed(r.nominal_cagr),
           format_pct_signed(r.post_tax_cagr),
           f"[{cagr_color(r.real_inr_cagr)}]{format_pct_signed(r.real_inr_cagr)}[/]",
           f"[{cagr_color(r.real_usd_cagr)}]{format_pct_signed(r.real_usd_cagr)}[/]",
       )

    console.print(table)


def render_erosion_strip(console: Console, portfolio: dict[str, float]) -> None:
    """A single horizontal bar attributing the gap between nominal and real USD.

    All three drag layers are positive (rates compound in the same direction),
    so they sum cleanly to the total drag. "Kept" is what the family ends up
    with in global purchasing power, shown separately because if it's
    negative the bar metaphor breaks.
    """
    r_n  = portfolio["nominal_cagr"]
    r_pt = portfolio["post_tax_cagr"]
    r_ri = portfolio["real_inr_cagr"]
    r_ru = portfolio["real_usd_cagr"]

    tax_pp = r_n - r_pt
    inf_pp = r_pt - r_ri
    fx_pp  = r_ri - r_ru
    total_drag = tax_pp + inf_pp + fx_pp
    if total_drag <= 0:
        return

    console.print(Text("§  EROSION OF NOMINAL CAGR", style="bold"))
    console.print(
        f"[dim]Nominal {format_pct_signed(r_n)} → "
        f"Real USD {format_pct_signed(r_ru)}.   "
        f"Total drag attributed below:[/]"
    )

    width = 50
    tax_w = max(1, round(width * tax_pp / total_drag))
    inf_w = max(1, round(width * inf_pp / total_drag))
    fx_w  = max(1, width - tax_w - inf_w)
    bar = (
        f"[red]{'█' * tax_w}[/]"
        f"[yellow]{'█' * inf_w}[/]"
        f"[magenta]{'█' * fx_w}[/]"
    )
    console.print(bar)

    legend = Table(box=None, padding=(0, 2), show_header=False)
    legend.add_column()
    legend.add_column(justify="right")
    legend.add_row("[red]■[/] Tax drag",          f"−{format_pct(tax_pp)}")
    legend.add_row("[yellow]■[/] Inflation drag", f"−{format_pct(inf_pp)}")
    legend.add_row("[magenta]■[/] FX drag",       f"−{format_pct(fx_pp)}")
    kept_color = "red" if r_ru < 0 else "green"
    legend.add_row(
        "[bold]Real USD CAGR (kept)[/]",
        f"[{kept_color} bold]{format_pct_signed(r_ru)}[/]",
    )
    console.print(legend)


def render_portfolio_summary(console: Console, portfolio: dict[str, float]) -> None:
    """The killer-line panel: portfolio CAGRs and absolute purchasing power.

    The second line is the line a wealth manager will quote to the client.
    """
    capital = portfolio["capital"]
    nominal_terminal = portfolio["nominal_terminal"]
    real_inr_terminal = portfolio["real_inr_terminal"]
    real_inr_gain = real_inr_terminal - capital
    horizon = portfolio["horizon"]

    line1 = (
        f"[bold]Portfolio CAGR[/]   "
        f"Nominal: [{cagr_color(portfolio['nominal_cagr'])}]"
        f"{format_pct_signed(portfolio['nominal_cagr'])}[/]   "
        f"Post-tax: [{cagr_color(portfolio['post_tax_cagr'])}]"
        f"{format_pct_signed(portfolio['post_tax_cagr'])}[/]   "
        f"Real INR: [{cagr_color(portfolio['real_inr_cagr'])}]"
        f"{format_pct_signed(portfolio['real_inr_cagr'])}[/]   "
        f"Real USD: [{cagr_color(portfolio['real_usd_cagr'])}]"
        f"{format_pct_signed(portfolio['real_usd_cagr'])}[/]"
    )
    line2 = (
        f"On {format_inr(capital)} over {horizon:.1f} years: "
        f"[bold]{format_inr(nominal_terminal)}[/] nominal · "
        f"[bold]{format_inr(real_inr_terminal)}[/] in today's purchasing power."
    )
    gain_color = "red" if real_inr_gain < 0 else "green"
    line3 = (
        f"[dim]Real wealth created (INR purchasing power):[/] "
        f"[{gain_color} bold]{format_inr(real_inr_gain)}[/] "
        f"[dim]over {horizon:.0f} years on {format_inr(capital)} of capital.[/]"
    )

    console.print(Panel(
        f"{line1}\n{line2}\n{line3}",
        title="[bold]§  PORTFOLIO VERDICT[/]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))


def render_inverted_footer(
    console: Console,
    results: list[WaterfallResult],
    cpi: float,
) -> None:
    """One line per asset: the macro condition under which it preserves wealth.

    Phrasing borrowed from TimeCell's own marketing copy — *"Show me the
    assumption you'd have to be wrong about for this to be the wrong call."*
    """
    console.print(Text("§  ASSUMPTIONS YOU'D HAVE TO BE WRONG ABOUT", style="bold"))
    console.print("[dim italic]For real INR return ≥ 0 (purchasing-power preservation):[/]")

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(width=2)
    table.add_column(no_wrap=True)
    table.add_column()

    for r in results:
        passes = r.breakeven_cpi >= cpi
        if passes:
            check = "[green]✓[/]"
            verdict = (
                f"Real-positive at any CPI ≤ "
                f"[green]{format_pct(r.breakeven_cpi)}[/]"
            )
        else:
            check = "[red]✗[/]"
            verdict = (
                f"CPI must fall to "
                f"[yellow]{format_pct(r.breakeven_cpi)}[/] "
                f"[dim](current: {format_pct(cpi)})[/]"
            )
        table.add_row(check, r.asset.name, verdict)

    console.print(table)


def render_footnote(console: Console) -> None:
    """The quiet professional sign-off."""
    console.print()
    console.print(
        "[dim]Asset CAGRs are indicative long-run averages "
        "(Nifty 50 TRI, S&P 500, RBI weighted FD, MCX gold, RBI HPI, BTC public).\n"
        "Tax rates assume HNI top slab; surcharge ignored. Positions assumed "
        "realised today.\n"
        "Structural illustration, not investment advice. Educational use only.[/]"
    )


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ════════════════════════════════════════════════════════════════════

def run_audit(
    assets: list[Asset],
    cpi: float = INDIA_CPI,
    fx: float = INR_USD_DEPRECIATION,
    console: Console | None = None,
) -> None:
    """End-to-end run: compute, aggregate, render."""
    console = console or Console()

    results = [
        compute_waterfall(a, NOMINAL_CAGR[a.asset_class], cpi, fx)
        for a in assets
    ]
    portfolio = aggregate_portfolio(results)

    console.print()
    render_header(console, cpi, fx)
    render_waterfall_table(console, results)
    console.print()
    render_erosion_strip(console, portfolio)
    console.print()
    render_portfolio_summary(console, portfolio)
    console.print()
    render_inverted_footer(console, results, cpi)
    render_footnote(console)
    console.print()


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Real Return Auditor — strip tax, inflation, and FX drag from "
            "portfolio returns to show what an Indian family actually keeps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--portfolio", type=Path,
        help="CSV with columns: name, asset_class, capital, years_held",
    )
    parser.add_argument(
        "--cpi", type=float, default=INDIA_CPI,
        help=f"India CPI as decimal (default: {INDIA_CPI})",
    )
    parser.add_argument(
        "--fx", type=float, default=INR_USD_DEPRECIATION,
        help=f"INR/USD depreciation as decimal (default: {INR_USD_DEPRECIATION})",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with a built-in sample HNI portfolio.",
    )
    args = parser.parse_args()

    console = Console()

    try:
        if args.portfolio:
            assets = load_portfolio_csv(args.portfolio)
        elif args.demo:
            assets = sample_portfolio()
        else:
            console.print(
                "[yellow]No portfolio supplied — running --demo. "
                "Use --portfolio FILE for your own.[/]"
            )
            assets = sample_portfolio()

        run_audit(assets, args.cpi, args.fx, console)
        return 0

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[bold red]Error:[/] {e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())