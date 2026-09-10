# Algua — Product Requirements

**Status:** vision of record, agreed 2026-09-06 between the operator (Lior) and the agent.
**Supersedes:** the thesis (§1) and roadmap (§9) of
`docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md`. That document's
architecture, walls and correctness essentials still stand; `docs/architecture.md` is the module
map. **Owner of changes:** the operator. An agent may propose an edit to this file by PR; it may
not merge one.

This is the document that says what Algua is *for*. When a spec, a plan, an issue or a review
disagrees with it, this document wins, and the disagreement is a bug in the other document.

---

## 1. What Algua is

Algua is a **proprietary trading operation run by agents and owned by Nix**. The product is
profit and loss on Nix's capital. The software is internal: it is not for sale, not for other
operators, and not multi-tenant.

Agents author strategies, run research, operate paper and live books, monitor the fleet and
repair the codebase. One human signs strategies live, decides how much capital is at stake, and
merges the code that guards those two things. Nothing else in the loop waits on a person.

## 2. Success

The headline number is **live P&L**, measured on the capital actually deployed, net of costs.

| Horizon | Success | Failure |
|---|---|---|
| Year one of live trading | Net positive P&L after costs on deployed capital | The book breaker trips (equity 15% below its high-water mark): the book flattens, every strategy returns to paper, and capital drops one rung (§3) |
| Year two onward | Annualized Sharpe of the live book above 1.0 | Sharpe below 0.5 over a rolling year: the operation is a research lab again and this document is reopened |

**The alpha check.** Once a year, and before any capital rung rises, the live book's daily returns
are regressed on SPY's. The intercept (alpha) must be positive at the 90% level. Beating SPY is
*not* the goal: a long equity book beats SPY in an up year by holding stocks, and a leveraged one
beats it every up year. The check exists so that the P&L we count is the P&L the machine claims
to produce, not the market's.

**Leading indicator, not success:** strategies reaching `forward_tested` per quarter. It measures
the factory, not the money.

## 3. Capital ladder

Capital is Nix's own. It rises on evidence, never on conviction, and never by more than one rung
at a time.

| Rung | Capital (draft) | Enter from below when |
|---|---|---|
| 0 — tuition | ~US$300 | Now. Money we have decided to lose while learning the surfaces. |
| 1 | ~US$40k (a quarter of the IBI withdrawal) | One live quarter (63 sessions) at rung 0 with net positive P&L, no breaker trip, and strategies that passed the **unrelaxed** forward gate (§8) |
| 2 | ~US$170k (all of it) | One live quarter at rung 1 with net positive P&L, no breaker trip, and a positive alpha check |

A breaker trip drops the ladder one rung automatically. Rising a rung is a human decision that
the ladder's conditions *permit*, never one it *requires*. Outside capital is a non-goal (§9).

The dollar figures are drafts; the rung structure is the requirement.

## 4. Thesis: where edge comes from

**Scale is the moat.** Algua tests as many uncorrelated hypotheses as it can, cheaply, fast and
honestly, and lets harsh forward selection keep the few that survive. No single market belief is
privileged.

Beliefs are still needed, or a factory produces noise. They live as **ideation categories** that
the idea engine rotates through, each with its own data requirements: institutional / whale flow
(the original 2026-05 thesis, now one category among several), cross-sectional momentum,
mean-reversion, seasonality, volatility structure, value / quality proxies, liquidity and
microstructure, event-driven, and whatever a new market lane (§5) makes possible.

The gates follow from the thesis and do not change per strategy kind or market:

- `backtested → candidate` is an **integrity floor**: point-in-time universe, delisting handling,
  costs on, a minimum holdout size, holdout Sharpe above zero, reproducible source. The whole
  statistical stack (deflated Sharpe, false-discovery ledger, breadth, regimes) computes and is
  recorded as advisory.
- `paper → forward_tested` is **the harsh threshold**: broker-clocked observations, realized
  Sharpe against a bar that overfit inflation raises on itself, drawdown and coverage bounds.
- `forward_tested → live` is the **human wall**: a signed challenge and a fresh forward
  certificate. No agent path exists and none will be added.

## 5. Markets, data, and time

**Four markets, as backtest data lanes now: US equities, crypto, forex, prediction markets.** A
live lane opens per market only when strategies on that market pass the forward gate in paper.
The cost of this decision is accepted up front: importers and a point-in-time store per market, a
24/7 calendar for markets that never close, and an instrument model for binary contracts with
resolution dates, all before any of those markets trades. Variety at the top of the funnel is the
bottleneck this decision removes.

**The data purchase.** Free providers reach roughly two years back; that starves both the
backtest gate (short holdouts) and the ideation categories (no regimes to test against). The
operator is buying history. The purchase should contain, in priority order:

1. US equities, survivorship-free, with delistings and corporate actions, twenty-plus years, at
   daily **and** one-minute resolution.
2. Crypto spot history at daily and minute resolution for the major pairs.
3. Forex majors at daily and minute resolution.
4. Whatever prediction-market history a vendor offers, with resolution outcomes.

Vendor choice is a cost question settled at purchase time; the FirstRate and Databento importers
already exist and any vendor's files land through the same importer seam.

**Two execution contracts.**

- **Daily**, the one that exists: decide on a closed session bar, fill no earlier than the next
  session's open. Crypto on this contract decides once per UTC day.
- **Intraday**, a second contract in its own right: decide on closed intraday bars at a fixed
  cadence (hourly, fifteen-minute), fill on the next bar, with its own clock for the tick loop,
  the reconcile, staleness, and the forward gate's observation count. It is a subsystem with its
  own spec, not a flag on the daily one.

Intraday *data* enters as features (realized volatility, VWAP, gaps) and as fill realism for
daily strategies as soon as it is imported; the intraday *contract* is step 7 of the roadmap.

## 6. What a strategy is

**One contract, four kinds.** A strategy is a pure function from point-in-time inputs to target
weights, optionally carrying a **versioned artifact**. The kinds differ only in what the artifact
is:

| Kind | Artifact | Rule that keeps it honest |
|---|---|---|
| Rule-based | none, or parameters | as today |
| Machine-learned | a fitted model | fitted point-in-time inside the walk-forward, the way the feature scaler already is; refit schedule is part of the strategy |
| Deep-learned | a trained network and its training window | same as above; training data provenance stamped like a data snapshot |
| Agentic | a prompt and a model identifier | sees only hindsight-safe inputs (the data wall applies to what the prompt is fed); every call is cached by input hash so a backtest is reproducible and a replay costs nothing |

Every kind passes the same gates (§4). No kind gets a shortcut, and no kind is excluded from the
forward gate because "the model already validated". The agentic kind is named as the
prompt-injection-to-live risk the 2026-05 spec anticipated; the human wall, the cache and the
data wall are the three defences.

Model artifacts, training windows and fitted parameters are stamped the way runs are stamped
today. The tracker backend that holds them is an open question (§11).

## 7. Division of labour

**The human is the wall and the wallet.** The human:

1. signs a strategy live;
2. moves capital between rungs;
3. funds the accounts;
4. merges code on CODEOWNERS-protected paths;
5. signs off the annual alpha check.

**Everything else is the agent's**: ideation, literature and web ingestion, authoring, research,
paper operation, live operation under the signed authorization, monitoring, incident response
within the walls, and repair of the codebase. This list of human duties **may not grow** without
a change to this document; every human step inside the funnel is a throughput cap that compounds
against the thesis.

**The operator builds the ideation system** (roadmap step 3); Eitan owns the knowledge-base
management workstream. The system is the agentic engine that produces hypotheses on a fixed
cadence, and the knowledge base it draws on. Nobody is a source of ideas themselves; the system is.

## 8. Tuition mode

The operator wants to start dirty: many strategies through the whole flow, the dashboard, the
monitoring, the signoff, on money that can be lost. The gates do not bend for this. Instead:

- At **rung 0 only**, the human may promote a strategy with signed relaxation flags on the
  research and forward gates. The signature binds the exact relaxation to the exact run, so a
  relaxed promotion is a recorded act of the human taking responsibility, never an agent path.
- Leaving rung 0 requires strategies that passed the **unrelaxed** forward gate. Tuition mode
  therefore ends by construction when the ladder climbs; nobody has to remember to stop.
- The operator's stated intent is to use the override to debug the flow, not to pick winners.

## 9. Non-goals

- **High-frequency or tick-level trading.** Bar-polling only; the intraday contract's floor is
  minutes, not microseconds.
- **Selling the software, hosting it for others, or multi-tenancy.**
- **Outside capital.** No second account, investor, or client; the accounting, reporting and
  regulatory picture stays that of one owner.
- **Humans curating ideas.** People point the ideation system at sources; they do not pick
  hypotheses.
- **Beating a benchmark as the goal.** SPY is the control in the alpha check, not the target.
- **A time-based one-time password as the live signoff.** The verifier secret would sit on the
  agent's machine. The signoff primitive is a signature by a key the agent never holds; a
  dashboard version of it is a passkey bound to the artifact hash.
- **Any agent path to `live`.**

## 10. Roadmap

Seven steps, in order. Each gets its own design spec and plan through the normal flow; this list
fixes intent and sequence, not implementation.

| # | Step | Unblocks |
|---|---|---|
| 1 | **Tuition trade to live**, plus the rung-0 rules: the signed override limited to rung 0, and the rung-1 entry condition | Every surface exercised end to end on real money; the ladder exists |
| 2 | **The data purchase and its import**: deep survivorship-free equities at daily and minute resolution | A backtest gate with statistical power; regimes to test categories against |
| 3 | **The ideation engine at a fixed cadence**, fed by literature and web ingestion | The top of the funnel the diagnosis found starved |
| 4 | **The model-artifact seam**: machine-learned, deep-learned and agentic strategy kinds under the one contract | Strategy variety; everything after is built against the seam |
| 5 | **New market data lanes**: crypto, forex, prediction markets as importers, calendars and instrument models | Market variety in the backtest funnel |
| 6 | **Crypto live lane** on the existing broker | A second live market at the lowest engineering cost |
| 7 | **The intraday execution contract** | Faster forward validation (63 hourly observations is two weeks) and the intraday categories |

The measure of the roadmap is the ladder (§3) and the year-one number (§2), not the number of
steps shipped.

## 11. Open questions

- **Tracker backend.** The MLflow file store Algua uses is deprecated upstream; model artifacts
  (§6) need a home before step 4.
- **Data vendor.** Settled at purchase (step 2); the importer seam is vendor-agnostic.
- **Rung dollar figures** (§3) are drafts pending the IBI withdrawal.
- **Prediction-market execution.** No broker integration is assumed; the instrument model comes
  first (step 5) and a live lane is a separate decision.

## 12. How this document is used

- `CLAUDE.md` points here first. An agent starting work reads this before the module map.
- A spec for any roadmap step cites the section it implements.
- A review may cite this document to refuse a change that contradicts it.
- The Nix venture note links here and does not copy it.
