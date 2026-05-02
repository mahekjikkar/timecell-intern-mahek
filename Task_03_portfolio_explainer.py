"""
AI-Powered Portfolio Explainer
==============================
Architecture:
    1. Validate input portfolio (Pydantic).
    2. Compute deterministic analytics in Python (the LLM never does math).
    3. Generate explanation via Gemini with strict JSON schema enforcement.
    4. Critique that explanation via a second, independent LLM call.
    5. Display raw + structured output for both stages.

Setup:
    pip install "google-genai>=0.3" "pydantic>=2.6"
    export GEMINI_API_KEY=your_key_here

Run:
    python portfolio_explainer.py
    # Optional: PORTFOLIO_TONE=beginner python portfolio_explainer.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from enum import Enum
from typing import Any, Final, Literal
import argparse
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator



# Model selection notes:
#   - gemini-2.5-flash: available on the free tier; good enough for both roles.

# Override at runtime with PORTFOLIO_PRIMARY_MODEL / PORTFOLIO_CRITIC_MODEL.
PRIMARY_MODEL: Final[str] = os.environ.get("PORTFOLIO_PRIMARY_MODEL", "gemini-2.5-flash")
CRITIC_MODEL:  Final[str] = os.environ.get("PORTFOLIO_CRITIC_MODEL",  "gemini-2.5-flash")

MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_SEC: Final[tuple[float, ...]] = (1.0, 3.0, 9.0)

TEMPERATURE_PRIMARY: Final[float] = 0.3   # some warmth in prose
TEMPERATURE_CRITIC: Final[float] = 0.1    # near-deterministic scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("portfolio_explainer")


# Domain enums and schemas

class Tone(str, Enum):
    BEGINNER = "beginner"
    EXPERIENCED = "experienced"
    EXPERT = "expert"


class Verdict(str, Enum):
    AGGRESSIVE = "Aggressive"
    BALANCED = "Balanced"
    CONSERVATIVE = "Conservative"


# ---- Input models -----------------------------------------------------------

class Asset(BaseModel):
    name: str
    allocation_pct: float = Field(ge=0, le=100)
    expected_crash_pct: float  # signed; negative = downside loss


class Portfolio(BaseModel):
    total_value_inr: float = Field(gt=0)
    monthly_expenses_inr: float = Field(gt=0)
    assets: list[Asset] = Field(min_length=1)

    @field_validator("assets")
    @classmethod
    def _allocations_sum_to_100(cls, v: list[Asset]) -> list[Asset]:
        total = sum(a.allocation_pct for a in v)
        if abs(total - 100.0) > 0.5:
            raise ValueError(
                f"Asset allocations must sum to 100% (±0.5); got {total:.2f}%"
            )
        return v


# ---- Pre-computed analytics (single source of truth for numbers) ------------

class DerivedAnalytics(BaseModel):
    weighted_stress_loss_pct: float
    weighted_stress_loss_inr: float
    emergency_fund_months: float
    top_concentration_pct: float
    top_concentration_asset: str
    equity_like_exposure_pct: float
    crypto_exposure_pct: float
    cash_pct: float

# ---- Primary LLM output schema ----------------------------------------------

class DoingWell(BaseModel):
    asset_referenced: str
    observation: str


class ConsiderChanging(BaseModel):
    asset_referenced: str
    observation: str
    rationale: str


class ExplanationOutput(BaseModel):
    risk_summary: str = Field(min_length=150, max_length=500)
    doing_well: DoingWell
    consider_changing: ConsiderChanging
    verdict: Verdict  


# ---- Critic LLM output schema -----------------------------------------------

class CritiqueIssue(BaseModel):
    axis: str
    severity: Literal["low", "medium", "high"]
    instruction: str


class CritiqueScores(BaseModel):
    numerical_accuracy: int = Field(ge=1, le=5)
    verdict_consistency: int = Field(ge=1, le=5)
    tone_match: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    regulatory_safety: int = Field(ge=1, le=5)


class CritiqueOutput(BaseModel):
    scores: CritiqueScores
    issues: list[CritiqueIssue]
    decision: Literal["PASS", "REVISE", "FAIL"]
    overall_comment: str


# Asset classification (lightweight heuristic)

_CATEGORY_KEYWORDS: Final[dict[str, set[str]]] = {
    "equity":      {"nifty", "sensex", "equity", "stock", "midcap", "smallcap",
                    "mutual fund", "elss", "index"},
    "crypto":      {"btc", "bitcoin", "eth", "ethereum", "crypto", "sol", "usdt"},
    "cash":        {"cash", "savings", "fd", "fixed deposit", "liquid fund",
                    "money market"},
    "debt":        {"bond", "debt", "ppf", "epf", "ncd", "g-sec", "gilt"},
    "commodity":   {"gold", "silver", "commodity", "sgb"},
    "real_estate": {"real estate", "property", "reit", "plot", "land"},
}


def classify_asset(name: str) -> str:
    """Map an asset name to a coarse category. Defaults to 'other'."""
    n = name.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(k in n for k in keywords):
            return category
    return "other"


# Analytics — pure, deterministic, never delegated to the LLM

def compute_analytics(portfolio: Portfolio) -> DerivedAnalytics:
    """Compute the figures the LLM is allowed to cite."""
    total = portfolio.total_value_inr
    monthly = portfolio.monthly_expenses_inr

    weighted_stress = 0.0
    equity_like = 0.0
    crypto = 0.0
    cash_pct = 0.0
    cash_value = 0.0

    top = portfolio.assets[0]
    for a in portfolio.assets:
        weighted_stress += (a.allocation_pct / 100.0) * abs(a.expected_crash_pct)
        category = classify_asset(a.name)

        if category == "equity":
            equity_like += a.allocation_pct
        elif category == "crypto":
            crypto += a.allocation_pct
            equity_like += a.allocation_pct  # crypto sits in equity-like risk bucket
        elif category == "cash":
            cash_pct += a.allocation_pct
            cash_value += (a.allocation_pct / 100.0) * total

        if a.allocation_pct > top.allocation_pct:
            top = a

    return DerivedAnalytics(
        weighted_stress_loss_pct=round(weighted_stress, 2),
        weighted_stress_loss_inr=round((weighted_stress / 100.0) * total, 2),
        emergency_fund_months=round(cash_value / monthly, 2) if monthly else 0.0,
        top_concentration_pct=round(top.allocation_pct, 2),
        top_concentration_asset=top.name,
        equity_like_exposure_pct=round(equity_like, 2),
        crypto_exposure_pct=round(crypto, 2),
        cash_pct=round(cash_pct, 2),
    )


# Prompt assembly

TONE_BLOCKS: Final[dict[Tone, str]] = {
    Tone.BEGINNER: (
        "Plain language. Avoid jargon; if a finance term is unavoidable, "
        "define it parenthetically. Reading level: smart 12th-grader. "
        "Short sentences. Reassuring but honest."
    ),
    Tone.EXPERIENCED: (
        "Standard finance vocabulary is fine (drawdown, concentration, "
        "duration, allocation). Skip definitions. Direct and concise. "
        "Mid-length sentences."
    ),
    Tone.EXPERT: (
        "Precise quantitative language. Reference stress-loss-weighted "
        "exposure, single-name concentration, illiquidity premium, etc. "
        "Terse and analytical. Assume CFA-level fluency."
    ),
}


PRIMARY_SYSTEM_TEMPLATE: Final[str] = """\
You are a portfolio analyst writing educational risk commentary for an Indian
wealth management platform serving HNW families. You are NOT a SEBI-registered
investment advisor. Your output is educational commentary on a portfolio the
investor has already assembled — never personalized investment advice.

CONTEXT
- All currency values are INR. Use lakh/crore notation (e.g., "₹1.2 crore").
- `expected_crash_pct` is a stress-test drawdown assumption per asset.
- Pre-computed analytics are provided. Use those numbers verbatim. Do NOT
  recompute, round differently, or invent additional figures.

TONE
{tone_block}

ANALYTICAL RULES
1. The verdict is determined by the rubric below. Apply it mechanically.
2. `doing_well.observation` must reference a specific asset, allocation, or
   behavior visible in the input — never generic praise.
3. `consider_changing` must (a) name the specific asset and (b) state the
   financial rationale (concentration, illiquidity, stress exposure, or
   expense-coverage gap).
4. `risk_summary` MUST cite at least one quantitative figure from the
   provided analytics.
5. Never recommend specific securities, fund names, ticker symbols, or
   buy/sell timing. Never use "should buy", "should sell", "I recommend".
   Use "the portfolio is exposed to…", "concentration in X is high
   relative to…", etc.


OUTPUT
Return ONE JSON object matching the provided schema. No prose before or
after. No markdown fences. No commentary about your reasoning.
"""


CRITIC_SYSTEM_PROMPT: Final[str] = """\
You are a senior financial editor reviewing draft portfolio commentary for
an Indian wealth platform. You evaluate; you do NOT rewrite.

You receive: portfolio, derived_analytics, intended persona, and the draft.
The derived_analytics is your source of truth for all numbers.

Score each axis 1–5 and produce structured findings.

AXES
1. NUMERICAL_ACCURACY — Do figures cited in the draft match
   derived_analytics? Flag any fabricated, mis-rounded, or contradicted number.
2. VERDICT_CONSISTENCY — Apply the rubric (Aggressive: stress >25 OR equity
   >70; Conservative: stress <10 AND equity <35; else Balanced). Does the
   draft's verdict match?
3. TONE_MATCH — Does language match the persona? Beginner drafts must not
   use undefined jargon; expert drafts must not over-explain basics.
4. SPECIFICITY — Are doing_well and consider_changing tied to a NAMED asset
   from the portfolio with a clear reason? Generic praise/criticism scores low.
5. REGULATORY_SAFETY — No specific securities, no buy/sell language, no
   implied advisor relationship.

For any axis scoring ≤ 3, `issues[]` MUST include a concrete, actionable
instruction. Bad: "Improve the tone." Good: "The phrase 'duration risk on
the debt sleeve' is too technical for a beginner — replace with 'risk that
bond prices fall when interest rates rise'."

DECISION
- PASS:   all axes ≥ 4
- REVISE: at least one axis = 3, none ≤ 2
- FAIL:   any axis ≤ 2

Return JSON only, matching the critique schema.
"""


def build_primary_system_prompt(tone: Tone) -> str:
    return PRIMARY_SYSTEM_TEMPLATE.format(tone_block=TONE_BLOCKS[tone])


def build_primary_user_prompt(
    portfolio: Portfolio,
    analytics: DerivedAnalytics,
    tone: Tone,
) -> str:
    return (
        "PORTFOLIO:\n"
        f"{portfolio.model_dump_json(indent=2)}\n\n"
        "DERIVED_ANALYTICS:\n"
        f"{analytics.model_dump_json(indent=2)}\n\n"
        f"INVESTOR_PERSONA: {tone.value}\n\n"
        "Produce the JSON now."
    )


def build_critic_user_prompt(
    portfolio: Portfolio,
    analytics: DerivedAnalytics,
    tone: Tone,
    draft: ExplanationOutput,
) -> str:
    return (
        "PORTFOLIO:\n"
        f"{portfolio.model_dump_json(indent=2)}\n\n"
        "DERIVED_ANALYTICS:\n"
        f"{analytics.model_dump_json(indent=2)}\n\n"
        f"INTENDED_PERSONA: {tone.value}\n\n"
        "DRAFT_TO_REVIEW:\n"
        f"{draft.model_dump_json(indent=2)}\n\n"
        "Produce the critique JSON now."
    )


# Gemini client wrapper

def _get_client() -> genai.Client:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in the environment."
        )
    return genai.Client()


def _is_permanent_quota_error(err: Exception) -> tuple[bool, str]:
    
    msg = str(err)
    low = msg.lower()
    if "limit: 0" in low:
        return True, (
            "The selected model is not available on your current Gemini tier "
            "(quota = 0). Switch to gemini-2.5-flash or enable billing."
        )
    if "perday" in low or "per_day" in low:
        return True, (
            "Daily quota exhausted for this model. It will not reset today. "
            "Switch models or wait ~24h."
        )
    return False, ""


def call_gemini(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: type[BaseModel],
    temperature: float,
) -> tuple[str, BaseModel]:
    
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    def _one_call(prompt: str) -> tuple[str, BaseModel | Exception]:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                raw = response.text or ""
                try:
                    return raw, response_schema.model_validate_json(raw)
                except (ValidationError, json.JSONDecodeError) as parse_err:
                    # Parse errors do NOT trigger network retry — surface them
                    # so the caller can decide whether to repair.
                    return raw, parse_err
            except genai_errors.APIError as api_err:
                # 429 with limit:0 or daily-exhausted → no point retrying.
                is_permanent, reason = _is_permanent_quota_error(api_err)
                if is_permanent:
                    raise RuntimeError(
                        f"Gemini quota exhausted (not retryable). {reason}\n"
                        f"Original error: {api_err}"
                    ) from api_err
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Gemini API failed after {MAX_RETRIES} attempts: {api_err}"
                    ) from api_err
                wait = RETRY_BACKOFF_SEC[attempt]
                log.warning(
                    "API error on attempt %d/%d: %s — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, api_err, wait,
                )
                time.sleep(wait)
        raise RuntimeError("Retry loop exited unexpectedly")  # pragma: no cover

    raw, result = _one_call(user_prompt)
    if isinstance(result, BaseModel):
        return raw, result

    # Schema/JSON parse failure → single repair attempt
    log.warning("Initial response did not validate. Attempting repair. Error: %s", result)
    repair_prompt = (
        f"{user_prompt}\n\n"
        f"Your previous response did not validate against the required schema.\n"
        f"Validation error: {result}\n"
        f"Return ONLY a corrected JSON object that exactly matches the schema. "
        f"No prose, no markdown fences."
    )
    raw, result = _one_call(repair_prompt)
    if isinstance(result, BaseModel):
        return raw, result
    raise RuntimeError(
        f"Schema validation failed after repair attempt.\n"
        f"Last error: {result}\n"
        f"Last raw output (truncated):\n{raw[:1000]}"
    )


# Pipeline functions

def generate_explanation(
    portfolio_dict: dict[str, Any],
    tone: str = "experienced",
) -> tuple[str, ExplanationOutput, DerivedAnalytics]:
    
    try:
        tone_enum = Tone(tone.lower())
    except ValueError as e:
        raise ValueError(
            f"Invalid tone '{tone}'. Choose from: {[t.value for t in Tone]}"
        ) from e

    try:
        portfolio = Portfolio.model_validate(portfolio_dict)
    except ValidationError as e:
        raise ValueError(f"Portfolio input failed validation:\n{e}") from e
    except (KeyError, TypeError) as e:
        raise ValueError(f"Portfolio input is malformed: {e}") from e

    analytics = compute_analytics(portfolio)
    log.info(
        "Analytics → stress=%.1f%% equity_like=%.1f%% top=%s@%.1f%% emergency_fund=%.1fmo",
        analytics.weighted_stress_loss_pct,
        analytics.equity_like_exposure_pct,
        analytics.top_concentration_asset,
        analytics.top_concentration_pct,
        analytics.emergency_fund_months,
    )

    sys_prompt = build_primary_system_prompt(tone_enum)
    usr_prompt = build_primary_user_prompt(portfolio, analytics, tone_enum)

    log.info("Calling primary model: %s (tone=%s)", PRIMARY_MODEL, tone_enum.value)
    raw, parsed = call_gemini(
        model=PRIMARY_MODEL,
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        response_schema=ExplanationOutput,
        temperature=TEMPERATURE_PRIMARY,
    )
    assert isinstance(parsed, ExplanationOutput)
    return raw, parsed, analytics


def critique_explanation(
    portfolio_dict: dict[str, Any],
    analytics: DerivedAnalytics,
    tone: str,
    draft: ExplanationOutput,
) -> tuple[str, CritiqueOutput]:
    """
    Secondary critique. Independent of the generator: receives the original
    portfolio + analytics so it can verify numbers from the source of truth.

    Returns:
        (raw_api_text, parsed_critique)
    """
    try:
        tone_enum = Tone(tone.lower())
    except ValueError as e:
        raise ValueError(f"Invalid tone '{tone}'") from e

    try:
        portfolio = Portfolio.model_validate(portfolio_dict)
    except (ValidationError, KeyError, TypeError) as e:
        raise ValueError(f"Portfolio input is malformed: {e}") from e

    user_prompt = build_critic_user_prompt(portfolio, analytics, tone_enum, draft)

    log.info("Calling critic model: %s", CRITIC_MODEL)
    raw, parsed = call_gemini(
        model=CRITIC_MODEL,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=CritiqueOutput,
        temperature=TEMPERATURE_CRITIC,
    )
    assert isinstance(parsed, CritiqueOutput)
    return raw, parsed


# Display

_BAR = "═" * 78
_DASH = "─" * 78


def _print_section(title: str) -> None:
    print(f"\n{_BAR}")
    print(f"  {title}")
    print(_BAR)


def print_raw_response(label: str, raw: str) -> None:
    _print_section(f"RAW API RESPONSE — {label}")
    print(raw if raw else "<empty response>")


def print_explanation(parsed: ExplanationOutput) -> None:
    _print_section("PARSED EXPLANATION")
    print(f"\nVerdict: {parsed.verdict.value}")
    print(f"\nRisk Summary:\n{parsed.risk_summary}")
    print(f"\nDoing Well — {parsed.doing_well.asset_referenced}")
    print(f"  {parsed.doing_well.observation}")
    print(f"\nConsider Changing — {parsed.consider_changing.asset_referenced}")
    print(f"  Observation: {parsed.consider_changing.observation}")
    print(f"  Rationale:   {parsed.consider_changing.rationale}")


def print_critique(critique: CritiqueOutput) -> None:
    _print_section("CRITIC REVIEW")
    s = critique.scores
    print("\nScores (1–5):")
    print(f"  Numerical Accuracy : {s.numerical_accuracy}")
    print(f"  Verdict Consistency: {s.verdict_consistency}")
    print(f"  Tone Match         : {s.tone_match}")
    print(f"  Specificity        : {s.specificity}")
    print(f"  Regulatory Safety  : {s.regulatory_safety}")
    print(f"\nDecision: {critique.decision}")
    print(f"Overall : {critique.overall_comment}")
    if critique.issues:
        print(f"\nIssues ({len(critique.issues)}):")
        for i, issue in enumerate(critique.issues, 1):
            print(f"  {i}. [{issue.severity.upper()}] {issue.axis}")
            print(f"     → {issue.instruction}")
    else:
        print("\nNo issues raised.")


# Test driver

PORTFOLIO_FIXTURE: Final[dict[str, Any]] = {
    "total_value_inr": 10_000_000,   # ₹1 crore
    "monthly_expenses_inr": 80_000,
    "assets": [
        {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
        {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
        {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
        {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct": 0},
    ],
}


def _load_portfolio(path_str: str | None) -> dict[str, Any]:
    """Load portfolio from JSON file, or fall back to the built-in fixture."""
    if path_str is None:
        log.info("No --portfolio provided. Using built-in fixture.")
        return PORTFOLIO_FIXTURE
    path = Path(path_str)
    if not path.is_file():
        raise ValueError(f"Portfolio file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Portfolio file is not valid JSON: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate AI-powered risk commentary for an HNW portfolio.",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Path to a portfolio JSON file. Defaults to the built-in fixture.",
    )
    parser.add_argument(
        "--tone",
        choices=[t.value for t in Tone],
        default=os.environ.get("PORTFOLIO_TONE", "experienced"),
        help="Explanation persona (default: $PORTFOLIO_TONE or 'experienced').",
    )
    args = parser.parse_args()

    try:
        portfolio_dict = _load_portfolio(args.portfolio)
    except ValueError as e:
        log.error("%s", e)
        return 2

    tone = args.tone
    log.info("Pipeline starting (tone=%s)", tone)

    # ---- Stage 1: Primary generation ----------------------------------------
    try:
        raw_primary, explanation, analytics = generate_explanation(
            portfolio_dict, tone=tone,
        )
    except ValueError as e:
        log.error("Input validation failed: %s", e)
        return 2
    except RuntimeError as e:
        log.error("Primary generation failed: %s", e)
        return 1

    print_raw_response("Primary Generator", raw_primary)
    print_explanation(explanation)

    # ---- Stage 2: Critique ---------------------------------------------------
    try:
        raw_critic, critique = critique_explanation(
            portfolio_dict, analytics, tone, explanation,
        )
    except RuntimeError as e:
        log.error("Critique failed (primary explanation still valid): %s", e)
        return 0

    print_raw_response("Critic", raw_critic)
    print_critique(critique)

    log.info("Pipeline complete. Critic decision: %s", critique.decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
