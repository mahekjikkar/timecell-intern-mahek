"""
Task_01_risk_metrics.py
=======================
Task 01: Portfolio Risk Metrics Calculator

Computes 5 required metrics + bonus moderate scenario + CLI bar chart.

"""


# ─── INPUT PORTFOLIO ──────────────────────────────────────────────────

portfolio = {
    "total_value_inr": 10_000_000,         # 1 Crore INR
    "monthly_expenses_inr": 80_000,
    "assets": [
        {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
        {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
        {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
        {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct": 0},
    ],
}


# ─── CORE FUNCTION ────────────────────────────────────────────────────

def _validate(portfolio: dict) -> None:
    assets = portfolio["assets"]
    total_alloc = sum(a["allocation_pct"] for a in assets)
    if not 99.5 <= total_alloc <= 100.5:
        raise ValueError(
            f"Allocations sum to {total_alloc}%, expected 100%. "
            f"Risk math is invalid otherwise."
        )
    for a in assets:
        if not -100 <= a["expected_crash_pct"] <= 0:
            raise ValueError(f"{a['name']}: crash_pct must be in [-100, 0]")
        if a["allocation_pct"] < 0:
            raise ValueError(f"{a['name']}: allocation_pct cannot be negative")

def compute_risk_metrics(portfolio: dict, crash_severity: float = 1.0) -> dict:
    _validate(portfolio)
    total_value = portfolio["total_value_inr"]
    monthly_exp = portfolio["monthly_expenses_inr"]
    assets      = portfolio["assets"]

    # ── 1. POST-CRASH VALUE ────────────────────────────────────────
    post_crash_value = 0
    for asset in assets:
        weight       = asset["allocation_pct"] / 100
        crash_pct    = asset["expected_crash_pct"] / 100 * crash_severity
        value_before = total_value * weight
        value_after  = value_before * (1 + crash_pct)
        post_crash_value += value_after

    # ── 2. RUNWAY MONTHS ───────────────────────────────────────────
    if monthly_exp <= 0:
        runway_months = float("inf")
    elif post_crash_value <= 0:
        runway_months = 0
    else:
        runway_months = post_crash_value / monthly_exp

    # ── 3. RUIN TEST ───────────────────────────────────────────────
    ruin_test = "PASS" if runway_months > 12 else "FAIL"

    # ── 4. LARGEST RISK ASSET ──────────────────────────────────────
    if not assets:
        largest_risk_asset = None
    else:
        largest_risk_asset = max(
            assets,
            key=lambda a: a["allocation_pct"] * abs(a["expected_crash_pct"]),
        )["name"]

    # ── 5. CONCENTRATION WARNING ───────────────────────────────────
    # True if ANY single asset is more than 40% of the portfolio.
    concentration_warning = any(a["allocation_pct"] > 40 for a in assets)

    return {
        "post_crash_value":      round(post_crash_value, 2),
        "runway_months":         round(runway_months, 2) if runway_months != float("inf") else float("inf"),
        "ruin_test":             ruin_test,
        "largest_risk_asset":    largest_risk_asset,
        "concentration_warning": concentration_warning,
    }


# ─── BONUS 1: COMPARE TWO SCENARIOS SIDE BY SIDE ──────────────────────

def compare_scenarios(portfolio: dict) -> None:
    """Print full crash vs moderate crash side by side."""
    full     = compute_risk_metrics(portfolio, crash_severity=1.0)
    moderate = compute_risk_metrics(portfolio, crash_severity=0.5)

    print("\n" + "=" * 65)
    print(f"   {'METRIC':<25} {'FULL CRASH':>17} {'MODERATE CRASH':>17}")
    print("=" * 65)

    rows = [
        ("Post-Crash Value (₹)", f"{full['post_crash_value']:>15,.0f}",
                                  f"{moderate['post_crash_value']:>15,.0f}"),
        ("Runway (months)",      f"{full['runway_months']:>15.1f}",
                                  f"{moderate['runway_months']:>15.1f}"),
        ("Ruin Test",            f"{full['ruin_test']:>15}",
                                  f"{moderate['ruin_test']:>15}"),
        ("Largest Risk Asset",   f"{full['largest_risk_asset']:>15}",
                                  f"{moderate['largest_risk_asset']:>15}"),
        ("Concentration Warning", f"{str(full['concentration_warning']):>15}",
                                   f"{str(moderate['concentration_warning']):>15}"),
    ]
    for label, val1, val2 in rows:
        print(f"   {label:<25} {val1:>17} {val2:>17}")
    print("=" * 65)


# ─── BONUS 2: CLI BAR CHART (no external libraries) ───────────────────

def plot_allocation_cli(portfolio: dict, bar_width: int = 40) -> None:
    """
    Print a simple CLI bar chart of the portfolio allocation.
    1 character ≈ (100 / bar_width) percent.
    """
    print("\n" + "=" * 65)
    print("   PORTFOLIO ALLOCATION")
    print("=" * 65)

    for asset in portfolio["assets"]:
        name  = asset["name"]
        pct   = asset["allocation_pct"]
        # Number of bar characters proportional to allocation
        bar   = "█" * int(pct / 100 * bar_width)
        print(f"   {name:<10} | {bar:<{bar_width}} {pct:>3}%")
    print("=" * 65)


# ─── MAIN ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required output: just the dictionary
    result = compute_risk_metrics(portfolio)

    print("\nRISK METRICS (Full Crash Scenario)")
    print("-" * 40)
    for key, value in result.items():
        print(f"  {key:<25} : {value}")

    # Bonus 1: side-by-side comparison
    compare_scenarios(portfolio)

    # Bonus 2: CLI allocation chart
    plot_allocation_cli(portfolio)
