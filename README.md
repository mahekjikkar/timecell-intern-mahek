# Timecell Intern — Technical Test Submission

A four-task submission for the Timecell.ai Engineering Intern (AI & Fintech) role.

| Task | What it does | Key skill | File |
|---|---|---|---|
| 01 | Portfolio risk metrics with full + moderate crash scenarios | Quantitative thinking, edge cases | `Task_01_risk_metrics.py` |
| 02 | Live market data feed (NIFTY, Reliance, BTC) with graceful failure | API plumbing, error handling | `Task_02_live_market_data.py` |
| 03 | LLM-powered portfolio explainer with critic-agent review | Prompt engineering, AI integration | `Task_03_portfolio_explainer.py` |
| 04 | "Real Return Auditor" — strips tax/CPI/FX from Indian portfolios | Initiative, judgment, taste | `Task_04_real_return_auditor.py` |

---

## Task 01 — Portfolio Risk Calculator

A pure-Python function that takes a portfolio dictionary and returns the five required risk metrics for a market crash scenario, plus a bonus moderate-crash comparison and a CLI bar chart.

### Math, in one place

For each asset, the post-crash value is `value_before × (1 + crash_pct × severity)`. The portfolio's post-crash value is the sum of these. Runway months = post-crash value / monthly expenses. Largest risk asset is the one with the highest `allocation_pct × |crash_pct|`. Concentration warning fires if any single asset exceeds 40%.

A `crash_severity` multiplier (1.0 = full crash, 0.5 = moderate) lets the same formula handle both scenarios with no duplicated logic.

### What I added beyond the spec

**Input validation (`_validate`).** Before any math runs, the function checks three things: allocations sum to 100% (±0.5 for rounding), each `expected_crash_pct` is within `[-100, 0]`, and no allocation is negative. The first check matters most — if a reviewer passes a portfolio whose allocations sum to 95%, the post-crash math silently pretends 5% of the capital doesn't exist. That's a real bug surface in wealth-management code, and validation catches it loudly instead of producing wrong numbers.

**Edge cases handled:** zero monthly expenses (returns `inf`), portfolio wiped out (returns 0 months), empty asset list (returns `None` for largest risk), 100% cash (post-crash equals starting value), the strict `>40%` boundary (40% exactly does not trigger).

### AI collaboration

Used Claude as a pair-programming collaborator. Specific decisions: rejected an early AI suggestion to pull in NumPy and matplotlib (over-engineered for the spec); manually wrote and re-checked the crash math; caught a bug where the concentration check was originally `>=`, but the spec says "more than 40%" which is strict `>`.

### Run

```bash
python Task_01_risk_metrics.py
```

No external dependencies. Standard library only.

---

## Task 02 — Live Market Data Fetch

A modular fetcher for three assets: NIFTY 50 (`^NSEI`), Reliance Industries (`RELIANCE.NS`), and Bitcoin (CoinGecko). Three-layer architecture — domain model, data fetching, presentation — so each concern is independently replaceable.

### Why these three

| Asset | Source | Why |
|---|---|---|
| NIFTY 50 | Yahoo Finance via `yfinance` | Free, no API key, `^` prefix routes to NSE index |
| Reliance Industries | Yahoo Finance via `yfinance` | `.NS` suffix pins to NSE exchange |
| Bitcoin | CoinGecko `/simple/price` | Free, unauthenticated, aggregates 900+ exchanges |

### Error handling — the contract

**Every fetcher always returns an `AssetQuote`. It never raises.** Exceptions are caught internally, logged via `RichHandler`, and returned as a degraded `AssetQuote` with `price=None` and a populated `error` field. This means a single API outage cannot interrupt the others — `fetch_all_quotes()` always returns three results, even if all three are failures.

Specifics:
- 10-second hard timeout on every HTTP call so a hung connection can't stall the script.
- Typed exception handling for CoinGecko: `Timeout`, `HTTPError` (with status code), and a generic catch-all are separated so log messages are operationally precise.
- Schema validation after HTTP 200 — a `None` price never silently passes downstream.
- Process exits with code `1` only if **all three** assets fail. Cron jobs and CI pipelines can detect a true outage.

### What I changed during review

Originally I had a hand-rolled IST timezone conversion (`UTC + timedelta(5:30)`) with a comment that "zoneinfo isn't required" — which was just wrong, since `zoneinfo` has been Python stdlib since 3.9. Replaced with `astimezone(ZoneInfo("Asia/Kolkata"))`. One line, correct, no hack.

### Run

```bash
pip install yfinance requests rich
python Task_02_live_market_data.py
```

Output is a Rich-rendered table of asset / symbol / current price / currency / fetch timestamp / status. Failed fetches appear as red `N/A` rows with details in a separate error panel below the table.

---

## Task 03 — AI-Powered Portfolio Explainer

A risk-commentary pipeline for HNW Indian portfolios. Takes a portfolio (built-in fixture or `--portfolio file.json`), produces a structured plain-English explanation via Gemini, then runs a second LLM pass that critiques the first across five axes.

### System design — separation of concerns

Five layers, each with one job:

- **Analytics layer (`compute_analytics`)** — pure Python that computes every number the LLM is allowed to cite: weighted stress loss, equity-like exposure, single-name concentration, emergency-fund months. **The LLM never does arithmetic.**
- **Prompt assembly (`build_*_prompt`)** — builds `(system_prompt, user_prompt)` from validated portfolio + analytics + persona. Templates are module-level constants, fully decoupled from API calls.
- **API client wrapper (`call_gemini`)** — handles structured-output config, exponential-backoff retries, fail-fast on permanent quota errors, and a one-shot repair attempt on schema validation failures.
- **Output contracts (Pydantic models)** — `ExplanationOutput` and `CritiqueOutput` define the response shape. Gemini's `response_schema` enforces server-side; `model_validate_json` re-validates client-side. Belt and suspenders.
- **Pipeline functions (`generate_explanation`, `critique_explanation`)** — pure orchestration. No prompt strings, no API plumbing.

A prompt change touches one constant. A schema change touches one Pydantic class. Nothing cascades.

### The Prompt Engineering Diary

The prompt went through several rounds before it was reliable.

**Attempt 1 — just ask for JSON.** Started with a basic "return JSON with these four fields" prompt. Mostly worked, but the model occasionally wrapped responses in ` ```json ` fences or led with a chatty intro like *"Sure! Here's the breakdown:"*. My parser broke on those.

**Attempt 2 — use structured outputs.** Switched to Gemini's `response_schema` config and passed my Pydantic model directly. The API now refuses to return anything that doesn't match the schema. Fences and preambles disappeared. I kept client-side `model_validate_json` as a safety net.

**Attempt 3 — fix the wrong numbers.** Even after JSON was reliable, the model would say *"around 35% stress loss"* when the real value was 43%. The fix wasn't a better prompt — it was to stop asking the model to do math at all. I wrote `compute_analytics()` that calculates everything beforehand, and the prompt injects it as a `DERIVED_ANALYTICS` block with the rule *"use these numbers verbatim."* After this, the figures in the output match exactly.

**Attempt 4 — the verdict consistency problem.** On borderline portfolios, the verdict (`Aggressive` / `Balanced` / `Conservative`) would flip between runs. Fix: spell out the rubric mechanically in the system prompt — *"Aggressive if stress > 25 OR equity > 70; Conservative if stress < 10 AND equity < 35; else Balanced"* — and have the critic check verdict consistency as one of its five axes.

**Attempt 5 — tone parameterization.** Added a `--tone` flag (`beginner` / `experienced` / `expert`) that injects different instruction blocks into the system prompt. Beginner gets short sentences and parenthetical jargon definitions. Expert gets terse quantitative language. Same facts, different bandwidth.

### The critic agent

The critic is a second LLM call with its own system prompt. It receives the original portfolio, the derived analytics (as ground truth for numbers), and the draft. It scores five axes 1–5:

1. **Numerical accuracy** — do figures cited in the draft match analytics?
2. **Verdict consistency** — does the verdict match the rubric?
3. **Tone match** — does language match the persona?
4. **Specificity** — are `doing_well` / `consider_changing` tied to a named asset?
5. **Regulatory safety** — no specific securities, no buy/sell language.

It returns a structured `CritiqueOutput` with per-axis scores, an `issues[]` array of severity-tagged actionable instructions, and a decision (`PASS` / `REVISE` / `FAIL`).

### Why the system prompt frames the LLM as "educational, not advisory"

> *"You are a portfolio analyst writing educational risk commentary for an Indian wealth management platform serving HNW families. You are NOT a SEBI-registered investment advisor."*

This framing does three things at once:
1. **Voice** — direct, honest commentary, not pitches.
2. **Specificity requirement** — `doing_well` and `consider_changing` must name a specific asset. Generic "your allocation is well diversified" is forbidden.
3. **Regulatory posture** — *"educational commentary on a portfolio the investor has already assembled"* is a different liability stance from *"personalized investment advice."* The banned-phrase list (no "should buy", "should sell", "I recommend") operationalizes that line.

### Run

```bash
pip install "google-genai>=0.3" "pydantic>=2.6"
export GEMINI_API_KEY=your_key_here

# Default fixture (BTC/NIFTY/GOLD/CASH from the spec)
python Task_03_portfolio_explainer.py

# Different portfolio + different tone
python Task_03_portfolio_explainer.py --portfolio sample_portfolio.json --tone beginner

# Expert tone
python Task_03_portfolio_explainer.py --portfolio sample_portfolio.json --tone expert
```

The script prints the raw Gemini response, a clean structured view of the parsed explanation, and the critic's review.

> **Note on Gemini quotas:** `gemini-2.5-pro` is paid-tier only on Google AI Studio. The defaults use `gemini-2.5-flash`, sufficient for both generator and critic on the free tier. Override via `PORTFOLIO_PRIMARY_MODEL` / `PORTFOLIO_CRITIC_MODEL`.

---

## Task 04 — The Real Return Auditor (Open Problem)

A terminal tool that strips four invisible layers of return erosion — Indian tax, CPI, INR-vs-USD currency drag, and the unrealized-vs-realized framing — to show what an Indian HNI family **actually keeps**.

### Why this

Every portfolio tracker shows one number: nominal return. Almost nothing shows what survives the four erosion layers that matter to an Indian HNI:

1. **Asset-class-asymmetric taxation.** A fixed deposit and a long-held equity SIP can show the same 8% headline. The FD pays slab rate (30%+ for HNIs); equity pays 12.5% LTCG (post-Budget-2024) above the ₹1.25L exemption. Same nominal, very different reality.
2. **CPI drag.** RBI's long-run band midpoint is 6%. A 7% FD often delivers a *negative* real return the moment it's marked to purchasing power.
3. **INR depreciation.** ~3–3.5% annually for two decades. A typical Indian portfolio is 90%+ INR-exposed. Foreign-education goals, overseas property theses, any global-spend liability — all are unhedged macro bets the family was never told they had.

The risk isn't a crash. It's the slow, invisible compounding of drag that a dashboard's headline number actively obscures. This tool is the question a tracker should have made you ask: *"what am I quietly losing every year, even when nothing crashes?"*

### Three deliberate design choices

Called out at the top of the file so a reviewer can challenge them:

1. **Hardcoded long-run class CAGRs**, not user-entered actual returns. The tool teaches structural truths about asset classes, not personal performance. Apples-to-apples.
2. **Positions assumed realised today.** Tax falls in full on accumulated gains. A "tax-deferred" framing would flatter every asset.
3. **Fisher relation, not subtraction.** Real return = `(1 + nominal) / (1 + drag) − 1`. Subtraction is a fine mental shortcut but breaks for large numbers.

### What the report shows

- **Per-asset waterfall** — Nominal → Post-tax → Real INR → Real USD CAGRs, color-coded.
- **Erosion strip** — a single horizontal bar attributing the gap between nominal and real-USD into tax / inflation / FX components.
- **Portfolio verdict** — capital-weighted CAGRs and absolute terminal values in both nominal INR and today's purchasing power.
- **Inverted-thinking footer** — for each asset, the assumption you'd have to be wrong about for it to preserve real wealth. *"For HDFC FD to break even, CPI must fall to 5.1% (current: 6.0%)."*

### Tax logic

Reflects the post-Budget-2024 Indian regime:

- Indian equity > 1y: 12.5% LTCG. ≤ 1y: 20% STCG.
- Foreign equity (LRS) > 2y: 12.5%. ≤ 2y: slab.
- Debt MF / FD: always slab (post-April-2023 regime — no indexation).
- Gold / Real estate > 2y: 12.5% LTCG, no indexation. ≤ 2y: slab.
- Crypto (VDA): flat 30%, regardless of horizon.

### Known simplifications

- HNI surcharge and 4% cess are ignored. True effective slab rate is closer to 35–43%. Override `HNI_SLAB_RATE` to model your bracket.
- US Equity (LRS) assumes contributions stay within the $250K/year LRS limit per individual.
- BTC's long-run CAGR is highly window-dependent. The 40% baseline is illustrative; override `NOMINAL_CAGR[CRYPTO]` for your preferred basis.

### Run

```bash
pip install rich

# Built-in HNI sample portfolio
python Task_04_real_return_auditor.py --demo

# Your own CSV
python Task_04_real_return_auditor.py --portfolio my_portfolio.csv

# Override macro assumptions
python Task_04_real_return_auditor.py --demo --cpi 0.065 --fx 0.04
```

CSV format: `name, asset_class, capital, years_held`. Asset classes: `equity_in`, `equity_us`, `debt_mf`, `fd`, `gold`, `real_estate`, `crypto`.

---

## What was hardest, and how I approached it

The hardest part was Task 3 — specifically, the moment I realised that fighting the LLM to produce correct numbers was the wrong fight. Two days in, I had a prompt with five rules about "calculate carefully" and "double-check your math," and the output was still occasionally wrong by 5–10 percentage points.

The unlock was inversion: the LLM doesn't have to do *anything* I can do deterministically in Python. I rebuilt around `compute_analytics()` as the single source of truth, made the prompt inject those numbers verbatim, and forbade recomputation. The same insight extended naturally to verdict-consistency (rubric in the prompt, critic verifies) and tone-stability (deterministic tone-block injection rather than relying on the model to "remember" the persona).

The general principle I came away with: **the LLM does prose, not arithmetic, not classification, not anything that has a clean Python expression.** Everything Python can compute deterministically should live in Python. The LLM is the last mile, not the engine.

---

## Submission

- **GitHub repo:** `timecell-intern-<your-name>`
- **Loom walkthrough:** [link]