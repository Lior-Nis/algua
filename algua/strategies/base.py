from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator

from algua.contracts.model_types import ModelHandle, ModelRef, compute_provenance_digest
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import ConstructFn, apply_capacity_cap

# The AUTHORED signal: a pure module-level `signal(view, params) -> pd.Series` of cross-sectional
# scores (NOT weights). The protocol-level `Strategy.target_weights(features)` is exposed only by
# the LoadedStrategy adapter, which composes signal -> construction.
SignalFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]

# OPTIONAL vectorized acceleration: a pure `signal_panel(bars, params)` returning the FULL decision-
# time SCORES matrix (index=timestamp, columns=symbol; PRE-lag) in one shot. NOT a second signal
# definition: the engine uses it only behind a fail-closed WEIGHT-level parity guard.
SignalPanelFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

# OPT-IN fundamentals signal (issue #132): `signal(view, params, fundamentals)`. Distinct type so
# the 2-arg and 3-arg forms never silently overload.
FundamentalsSignalFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]

# OPT-IN news signal (issue #132): `signal(view, params, news)`. Distinct type so the news 3-arg
# form never silently overloads the 2-arg or fundamentals 3-arg forms.
NewsSignalFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]

# OPT-IN model signal (issue #376): `signal(view, params, model)` where `model` is a ModelHandle
# (resolved artifact bytes + immutable ModelVersion) injected by the loader. Distinct type so the
# model 3-arg form never silently overloads the fundamentals/news 3-arg forms. Unlike those lanes,
# the model is FIXED for the whole run (bound once at load), so no per-bar sidecar is injected.
ModelSignalFn = Callable[[pd.DataFrame, dict[str, Any], ModelHandle], pd.Series]


class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}
    # Portfolio-construction policy (issue #141): the id is resolved by the loader against
    # algua.portfolio.construction; construction_params are validated per-policy at load.
    construction: str
    construction_params: dict[str, Any] = {}
    # Opt into the as-of fundamentals lane (issue #132). When True the loader binds the 3-arg
    # signal as `fundamentals_signal_fn` and the engine injects the PIT-correct frame per bar.
    needs_fundamentals: bool = False
    # Opt into the as-of news lane (issue #132). Mutually exclusive with needs_fundamentals.
    needs_news: bool = False
    # Opt into the model lane (issue #376). Mutually exclusive with needs_fundamentals/needs_news.
    # When True the loader resolves `model_ref` against the model registry and binds the 3-arg
    # signal as `model_signal_fn`, injecting the resolved ModelHandle.
    needs_model: bool = False
    # PINNED model reference — required iff needs_model, forbidden otherwise. NEVER 'latest': an
    # explicit version + the artifact digest + training_as_of cutoff + provenance digest the config
    # was validated against, so a strategy can never silently bind a different model.
    model_ref: ModelRef | None = None
    # Declared maximum feature lookback in bars (issue #345): how many trailing bars the signal
    # reads to score a single bar (e.g. a 60-bar trailing-return momentum => 60). Drives the
    # walk-forward train/holdout embargo (purge gap = max(feature_lookback, decision_lag_bars)) so
    # the in-sample selection stats can't share feature/decision windows with the holdout.
    # `None` = UNDECLARED (the agent `research promote` path fails closed on it — declare it, even
    # to 0 for a strategy with no rolling feature window). An explicit value (incl. 0) is honored.
    # Author contract: declare >= the largest lookback the signal will ever use, including any
    # value you intend to sweep (under-declaration is an author bug, like a wrong `universe`).
    feature_lookback: int | None = None

    @field_validator("feature_lookback")
    @classmethod
    def _non_negative_lookback(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("feature_lookback must be >= 0 (or None if undeclared)")
        return v

    @field_validator("execution", mode="before")
    @classmethod
    def _execution_must_be_a_contract(cls, v: object) -> object:
        # `execution` must be an already-constructed ExecutionContract (which every strategy builds
        # directly). A raw mapping would let pydantic coerce nested values BEFORE the dataclass
        # __post_init__ guards run — e.g. bool `True` -> 1.0 for reference_aum / max_participation_
        # rate (fail-OPEN capacity, #344), or "false" -> allow_short. Reject it: fail closed at the
        # boundary so those __post_init__ rails can never be bypassed via dict input.
        if isinstance(v, Mapping):
            raise ValueError(
                "execution must be an ExecutionContract instance, not a raw mapping (a dict "
                "bypasses ExecutionContract/CapacityLimit __post_init__ rails via coercion)"
            )
        return v


@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + the authored signal fn(s) + the RESOLVED construction policy into an
    object satisfying the Strategy protocol. Exactly one of (`signal_fn`, `fundamentals_signal_fn`,
    `news_signal_fn`) is active, selected by the `needs_fundamentals`/`needs_news` flags (mutually
    exclusive). The adapter is the ONLY place the protocol-level `target_weights` exists; it
    composes construct(signal(view), view).

    `construct_fn` is the RAW policy callable (never a params-bound partial): `construct` reads
    `config.construction_params` at call time, so a sweep that rebuilds the config takes effect and
    `inspect.getmodule(construct_fn)` resolves to the policy module for the identity hash.
    """

    config: StrategyConfig
    construct_fn: ConstructFn
    signal_fn: SignalFn | None = None
    signal_panel_fn: SignalPanelFn | None = None
    fundamentals_signal_fn: FundamentalsSignalFn | None = None
    news_signal_fn: NewsSignalFn | None = None
    model_signal_fn: ModelSignalFn | None = None
    # The model artifact bound at load time for a needs_model strategy (issue #376) — fixed for the
    # whole run (PIT-safe), injected as the 3rd arg to `model_signal_fn`.
    model_handle: ModelHandle | None = None

    def __post_init__(self) -> None:
        cfg = self.config
        # Three-way (fundamentals / news / model) exclusivity — a strategy uses exactly one PIT
        # sidecar lane, or none.
        exclusive = [
            n for n, on in (
                ("needs_fundamentals", cfg.needs_fundamentals),
                ("needs_news", cfg.needs_news),
                ("needs_model", cfg.needs_model),
            ) if on
        ]
        if len(exclusive) > 1:
            raise ValueError(
                f"at most one of needs_fundamentals/needs_news/needs_model may be True; "
                f"got {exclusive} (a strategy combining lanes is not supported yet)"
            )
        if cfg.needs_model and cfg.model_ref is None:
            raise ValueError("needs_model=True requires model_ref to be set")
        if not cfg.needs_model and cfg.model_ref is not None:
            raise ValueError("model_ref set but needs_model is False (forbidden)")
        if cfg.needs_model and self.model_handle is None:
            raise ValueError("needs_model=True requires a resolved model_handle")
        # The bound handle MUST be the pinned model — otherwise a caller could pass a safe old
        # model_ref (which the engine PIT-guards) while binding a different (e.g. future-trained)
        # artifact that signal() would actually use. Enforce identity here, at construction, so no
        # construction path (loader OR a direct programmatic build) can diverge (#376).
        if cfg.needs_model:
            assert cfg.model_ref is not None
            assert self.model_handle is not None
            bound = self.model_handle.version
            if (
                bound.name != cfg.model_ref.name
                or bound.version != cfg.model_ref.version
                or bound.digest != cfg.model_ref.digest
                or bound.training_as_of != cfg.model_ref.training_as_of
                or bound.provenance_digest != cfg.model_ref.provenance_digest
            ):
                raise ValueError(
                    "bound model_handle does not match the pinned model_ref "
                    "(name/version/digest/training_as_of/provenance) — refusing to bind a "
                    "different model than the config was validated against"
                )
            # The metadata matching the pin is not enough: verify the ACTUAL artifact bytes hash to
            # the pinned digest, and that the version's metadata reproduces its own provenance
            # digest. Otherwise a direct programmatic build could present matching ModelVersion
            # metadata while binding arbitrary (e.g. future-trained) bytes that signal() consumes —
            # the exact silent-different-model / PIT bypass we fail closed on (#376).
            bytes_digest = hashlib.sha256(self.model_handle.artifact_bytes).hexdigest()[:16]
            if bytes_digest != cfg.model_ref.digest:
                raise ValueError(
                    f"bound model artifact bytes (digest {bytes_digest}) do not hash to the pinned "
                    f"model_ref.digest {cfg.model_ref.digest} — refusing to bind a different model"
                )
            expected_prov = compute_provenance_digest(
                digest=bound.digest,
                training_snapshot_id=bound.training_snapshot_id,
                training_as_of=bound.training_as_of,
                code_hash=bound.code_hash,
                hyperparameters=bound.hyperparameters,
                seed=bound.seed,
                eval_report=bound.eval_report,
            )
            if expected_prov != cfg.model_ref.provenance_digest:
                raise ValueError(
                    "bound model_handle provenance does not reproduce the pinned "
                    "provenance_digest — the version metadata was forged; refusing to bind"
                )
        decision_fns = {
            "signal_fn": self.signal_fn,
            "fundamentals_signal_fn": self.fundamentals_signal_fn,
            "news_signal_fn": self.news_signal_fn,
            "model_signal_fn": self.model_signal_fn,
        }
        active = [k for k, v in decision_fns.items() if v is not None]
        expected = (
            "fundamentals_signal_fn" if cfg.needs_fundamentals
            else "news_signal_fn" if cfg.needs_news
            else "model_signal_fn" if cfg.needs_model
            else "signal_fn"
        )
        if active != [expected]:
            raise ValueError(
                f"config requires exactly {expected!r} to be set "
                f"(needs_fundamentals={cfg.needs_fundamentals}, needs_news={cfg.needs_news}, "
                f"needs_model={cfg.needs_model}); got {active}"
            )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def universe(self) -> list[str]:
        return self.config.universe

    @property
    def execution(self) -> ExecutionContract:
        return self.config.execution

    @property
    def params(self) -> dict[str, Any]:
        return self.config.params

    @property
    def authored_signal(self) -> SignalFn | FundamentalsSignalFn | NewsSignalFn | ModelSignalFn:
        """The active authored signal fn — used wherever code needs the strategy's source module
        (e.g. code_hash), since signal_fn is None for a needs_fundamentals/needs_news/needs_model
        strategy."""
        if self.config.needs_fundamentals:
            assert self.fundamentals_signal_fn is not None
            return self.fundamentals_signal_fn
        if self.config.needs_news:
            assert self.news_signal_fn is not None
            return self.news_signal_fn
        if self.config.needs_model:
            assert self.model_signal_fn is not None
            return self.model_signal_fn
        assert self.signal_fn is not None
        return self.signal_fn

    def signal(
        self,
        view: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.Series:
        if self.config.needs_fundamentals:
            if fundamentals is None:
                raise ValueError(
                    f"strategy {self.name!r} needs fundamentals but signal was called without a "
                    f"fundamentals frame (fail closed)"
                )
            if news is not None:
                raise ValueError(f"strategy {self.name!r} was passed a news frame it does not use")
            assert self.fundamentals_signal_fn is not None
            return self.fundamentals_signal_fn(view, self.config.params, fundamentals)
        if self.config.needs_news:
            if news is None:
                raise ValueError(
                    f"strategy {self.name!r} needs news but signal was called without a news "
                    f"frame (fail closed)"
                )
            if fundamentals is not None:
                raise ValueError(
                    f"strategy {self.name!r} was passed a fundamentals frame it does not use"
                )
            assert self.news_signal_fn is not None
            return self.news_signal_fn(view, self.config.params, news)
        if self.config.needs_model:
            if fundamentals is not None or news is not None:
                raise ValueError(
                    f"strategy {self.name!r} is a model strategy but was passed a "
                    f"fundamentals/news frame it does not use"
                )
            assert self.model_signal_fn is not None
            assert self.model_handle is not None
            return self.model_signal_fn(view, self.config.params, self.model_handle)
        if fundamentals is not None or news is not None:
            raise ValueError(f"strategy {self.name!r} takes no PIT sidecar but one was passed")
        assert self.signal_fn is not None
        return self.signal_fn(view, self.config.params)

    def signal_panel(self, bars: pd.DataFrame) -> pd.DataFrame | None:
        if self.signal_panel_fn is None:
            return None
        return self.signal_panel_fn(bars, self.config.params)

    def construct(self, scores: pd.Series, view: pd.DataFrame) -> pd.Series:
        weights = self.construct_fn(scores, view, self.config.construction_params)
        # ADV / capacity participation cap (issue #344). Applied HERE — the single chokepoint every
        # path (backtest loop, vectorized fast path + its parity twin, live/paper decide) resolves
        # weights through — so the cap is enforced identically everywhere. `view` is the same PIT
        # frame the signal saw (ends at the fully-closed decision bar t), so the trailing ADV never
        # sees the fill bar. No-op when no capacity budget is declared.
        capacity = self.config.execution.capacity
        if capacity is not None:
            weights = apply_capacity_cap(weights, view, capacity)
        return weights

    def target_weights(
        self,
        features: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.Series:
        return self.construct(self.signal(features, fundamentals, news), features)


def assert_tradable_without_fundamentals(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_fundamentals strategy must NOT run paper/live yet — the as-of
    fundamentals lane is wired only into the backtest engine (issue #132). Called at every trading
    load point so no actor (agent promote OR human raw transition) can run it blind."""
    if strategy.config.needs_fundamentals:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_fundamentals; paper/live fundamentals "
            f"wiring is not built yet (#132 follow-up) — refusing to trade it blind"
        )


def assert_tradable_without_news(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_news strategy must NOT run paper/live yet — the as-of news lane is
    wired only into the backtest engine (issue #132). Called at every trading load point."""
    if strategy.config.needs_news:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_news; paper/live news wiring is not built "
            f"yet (#132 follow-up) — refusing to trade it blind"
        )


def assert_tradable_without_model(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_model strategy must NOT run paper/live yet — the model lane is wired
    only into the `backtest run` engine (issue #376). Called at every trading load point."""
    if strategy.config.needs_model:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_model; paper/live model wiring is not "
            f"built yet (#376 follow-up) — refusing to trade it blind"
        )


def config_hash(strategy: LoadedStrategy) -> str:
    """Stable digest of a strategy's resolved configuration (name + universe + params +
    execution contract + construction policy id and params). The single source of truth for the
    config side of the artifact identity, shared by the backtest engine and the registry's
    live-approval gate. The declared `feature_lookback` (#345) is folded in too, since it sizes the
    walk-forward embargo and therefore the carved windows.

    Serializes the *full* ExecutionContract via asdict, so every behavior-affecting field
    (warmup_bars, allow_fractional, max_gross_exposure, decision_lag_bars, rebalance_frequency)
    is part of the identity — and any field added later is included automatically. The construction
    policy id + its params (issue #141) are folded in too, so retuning the construction policy
    (e.g. top_k) or swapping the policy invalidates a prior live approval. Serialized with
    allow_nan=False so a non-finite construction param cannot produce a non-canonical hash. The
    digest is a 128-bit sha256 prefix (#341): collision-resistant for identity/gate use, not a
    collision-proof guarantee — two differing configs are astronomically unlikely, not provably
    unable, to collide."""
    identity: dict[str, Any] = {
        "name": strategy.name,
        "universe": strategy.universe,
        "params": strategy.params,
        "execution": asdict(strategy.execution),
        "construction": strategy.config.construction,
        "construction_params": strategy.config.construction_params,
        "needs_fundamentals": strategy.config.needs_fundamentals,
        "needs_news": strategy.config.needs_news,
        # #345: behavior-affecting (sizes the walk-forward embargo), and NOT inside params /
        # execution, so it must be folded in explicitly — two runs with different declared
        # lookbacks carve different windows and must never collide on config_hash.
        "feature_lookback": strategy.config.feature_lookback,
    }
    # Model identity (issue #376) is folded in ONLY when needs_model is True, so every existing
    # non-model strategy's config_hash is byte-identical to before (no live-approval / result-
    # identity churn). The pinned model_ref carries name/version/digest/training_as_of AND the
    # provenance_digest (which commits to the training snapshot/code/hyperparameters/seed/eval
    # report), so the full training provenance — not just the version — is part of the identity.
    if strategy.config.needs_model:
        assert strategy.config.model_ref is not None
        identity["needs_model"] = True
        identity["model_ref"] = strategy.config.model_ref.as_dict()
    payload = json.dumps(identity, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
