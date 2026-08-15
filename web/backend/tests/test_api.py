"""API surface: httpx ASGITransport against the app, run_cli monkeypatched."""

from __future__ import annotations

from typing import Any

import backend.main as main_mod
import httpx
import pytest
from backend.algua_cli import CliError

FETCHED_AT = "2026-08-09T00:00:00+00:00"


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=main_mod.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _env(data: Any, fetched_at: str = FETCHED_AT, stale: bool = False) -> dict[str, Any]:
    return {"ok": True, "data": data, "fetched_at": fetched_at, "stale": stale}


def _route_cli(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[tuple[str, ...], Any],
) -> list[tuple[str, ...]]:
    """Monkeypatch backend.main's run_cli with a per-argv dict router.

    A route value that is an Exception is raised; anything else is returned.
    Returns the list of argv tuples the fake received, in call order.
    """
    seen: list[tuple[str, ...]] = []

    async def fake_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        seen.append(args)
        result = routes[args]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    return seen


# --- /api/fleet (slice B) ---


async def test_fleet_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = {
        "ok": True,
        "data": {"ok": False, "summary": {"alerting": 0}, "rows": []},
        "fetched_at": FETCHED_AT,
        "stale": False,
    }
    seen: dict[str, Any] = {}

    async def fake_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        seen["args"] = args
        seen["ttl_s"] = ttl_s
        return envelope

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    async with _client() as client:
        resp = await client.get("/api/fleet")
    assert resp.status_code == 200
    assert resp.json() == envelope
    assert seen["args"] == ("fleet", "health")
    assert seen["ttl_s"] == 10.0


async def test_fleet_cli_error_is_502_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        raise CliError("cli_timeout", "algua fleet health exceeded 60.0s wall timeout")

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    async with _client() as client:
        resp = await client.get("/api/fleet")
    assert resp.status_code == 502
    assert resp.json() == {
        "ok": False,
        "error": "algua fleet health exceeded 60.0s wall timeout",
        "code": "cli_timeout",
    }


async def test_healthz() -> None:
    async with _client() as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# --- /api/strategies ---


async def test_strategies_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    # registry list emits a bare array; the seam wraps it as {"data": [...]}.
    envelope = _env({"data": [{"name": "momo", "stage": "paper"}]})
    seen: dict[str, Any] = {}

    async def fake_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        seen["args"] = args
        seen["ttl_s"] = ttl_s
        return envelope

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    async with _client() as client:
        resp = await client.get("/api/strategies")
    assert resp.status_code == 200
    assert resp.json() == envelope
    assert seen["args"] == ("registry", "list")
    assert seen["ttl_s"] == 60.0


# --- /api/strategy/{name} composition ---

_REGISTRY_DATA = {"name": "momo", "stage": "paper", "transitions": []}
_PAPER_DATA = {"strategy": "momo", "health": "ok", "recent_orders": []}
_GATES_DATA = {"strategy": "momo", "gate_evaluations": [], "forward_gate_evaluations": []}


def _gates_argv(name: str) -> tuple[str, ...]:
    """The gates argv the app actually issues — the limit is EXPLICIT so the composed
    response can tell the UI a saturated ledger is truncated, not complete."""
    return ("registry", "gates", name, f"--limit={main_mod.GATE_HISTORY_LIMIT}")


def _strategy_routes(**overrides: Any) -> dict[tuple[str, ...], Any]:
    routes: dict[tuple[str, ...], Any] = {
        ("registry", "show", "momo"): _env(_REGISTRY_DATA, fetched_at="2026-08-09T00:00:02+00:00"),
        ("paper", "show", "momo"): _env(_PAPER_DATA, fetched_at="2026-08-09T00:00:01+00:00"),
        _gates_argv("momo"): _env(_GATES_DATA, fetched_at="2026-08-09T00:00:03+00:00"),
    }
    routes.update({k: v for k, v in overrides.items()})  # type: ignore[misc]
    return routes


async def test_strategy_composition_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _route_cli(monkeypatch, _strategy_routes())
    async with _client() as client:
        resp = await client.get("/api/strategy/momo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["strategy"] == "momo"
    assert body["registry"] == _REGISTRY_DATA
    assert body["paper"] == _PAPER_DATA
    assert body["gates"] == _GATES_DATA
    assert body["fetched_at"] == "2026-08-09T00:00:01+00:00"  # min of the parts
    assert body["stale"] is False
    assert "part_errors" not in body
    # The UI needs the requested per-ledger limit to tell a saturated ledger from a
    # complete history.
    assert body["gates_limit"] == main_mod.GATE_HISTORY_LIMIT


async def test_strategy_paper_part_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _strategy_routes()
    routes[("paper", "show", "momo")] = CliError("cli_timeout", "paper show timed out")
    _route_cli(monkeypatch, routes)
    async with _client() as client:
        resp = await client.get("/api/strategy/momo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["registry"] == _REGISTRY_DATA
    assert body["paper"] is None
    assert body["gates"] == _GATES_DATA
    assert body["part_errors"] == {"paper": "cli_timeout"}


async def test_strategy_registry_not_found_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    err = CliError("not_found", "strategy 'ghost' not found")
    _route_cli(
        monkeypatch,
        {
            ("registry", "show", "ghost"): err,
            ("paper", "show", "ghost"): err,
            _gates_argv("ghost"): err,
        },
    )
    async with _client() as client:
        resp = await client.get("/api/strategy/ghost")
    assert resp.status_code == 404
    assert resp.json() == {
        "ok": False,
        "error": "strategy 'ghost' not found",
        "code": "not_found",
    }


async def test_strategy_registry_other_cli_error_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _strategy_routes()
    routes[("registry", "show", "momo")] = CliError("cli_timeout", "registry show timed out")
    _route_cli(monkeypatch, routes)
    async with _client() as client:
        resp = await client.get("/api/strategy/momo")
    assert resp.status_code == 502
    assert resp.json() == {
        "ok": False,
        "error": "registry show timed out",
        "code": "cli_timeout",
    }


async def test_strategy_stale_part_ors_into_composed_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _strategy_routes()
    routes[_gates_argv("momo")] = _env(_GATES_DATA, stale=True)
    _route_cli(monkeypatch, routes)
    async with _client() as client:
        resp = await client.get("/api/strategy/momo")
    assert resp.status_code == 200
    assert resp.json()["stale"] is True


async def test_strategy_name_leading_dash_is_422() -> None:
    async with _client() as client:
        resp = await client.get("/api/strategy/-momo")
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["code"] == "invalid_param"


# --- /api/strategy/{name}/series ---


async def test_series_passes_since_and_lane_as_single_argv_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _env(
        {
            "strategy": "momo",
            "lane_filter": "paper",
            "series": {"paper": [], "live": []},
            "truncated": False,
        }
    )
    expected_argv = (
        "fleet",
        "series",
        "momo",
        "--since=2026-08-01T00:00:00+00:00",
        "--lane=paper",
    )
    seen = _route_cli(monkeypatch, {expected_argv: envelope})
    async with _client() as client:
        resp = await client.get(
            "/api/strategy/momo/series",
            params={"since": "2026-08-01T00:00:00+00:00", "lane": "paper"},
        )
    assert resp.status_code == 200
    assert resp.json() == envelope
    assert seen == [expected_argv]


async def test_series_no_params_passes_bare_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = _env({"strategy": "momo", "lane_filter": None, "series": {"paper": [], "live": []}})
    seen = _route_cli(monkeypatch, {("fleet", "series", "momo"): envelope})
    async with _client() as client:
        resp = await client.get("/api/strategy/momo/series")
    assert resp.status_code == 200
    assert seen == [("fleet", "series", "momo")]


@pytest.mark.parametrize(
    "path,params",
    [
        ("/api/strategy/momo/series", {"lane": "margin"}),  # bad lane
        ("/api/strategy/momo/series", {"since": "not-a-date"}),  # bad since
        ("/api/strategy/-momo/series", {}),  # name with leading dash
        ("/api/strategy/mo=mo/series", {}),  # name with a char outside the allowlist
    ],
)
async def test_series_param_validation_is_422(
    monkeypatch: pytest.MonkeyPatch, path: str, params: dict[str, str]
) -> None:
    async def fail_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        raise AssertionError(f"run_cli must not be reached, got argv {args!r}")

    monkeypatch.setattr(main_mod, "run_cli", fail_run_cli)
    async with _client() as client:
        resp = await client.get(path, params=params)
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["code"] == "invalid_param"


async def test_series_name_with_slash_never_reaches_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An encoded slash decodes to a path that no longer matches the route -> 404,
    # so a slashed name can never reach the CLI; the validator itself is pinned
    # against slashes in test_params.py.
    async def fail_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        raise AssertionError(f"run_cli must not be reached, got argv {args!r}")

    monkeypatch.setattr(main_mod, "run_cli", fail_run_cli)
    async with _client() as client:
        resp = await client.get("/api/strategy/mo%2Fmo/series")
    assert resp.status_code in (404, 422)


# --- /api/activity ---


async def test_activity_uses_flag_equals_value_argv_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _env({"data": [{"action": "transition", "actor": "agent"}]})
    expected_argv = (
        "audit",
        "log",
        "--limit=250",
        "--strategy=momo",
        "--actor=agent",
        "--action=transition",
    )
    seen = _route_cli(monkeypatch, {expected_argv: envelope})
    async with _client() as client:
        resp = await client.get(
            "/api/activity",
            params={"strategy": "momo", "actor": "agent", "action": "transition", "limit": 250},
        )
    assert resp.status_code == 200
    assert resp.json() == envelope
    assert seen == [expected_argv]
    # The reviewed injection defense: each option is ONE argv element in --flag=value form.
    assert "--actor=agent" in seen[0]
    assert "--strategy=momo" in seen[0]
    assert "--action=transition" in seen[0]
    assert "--limit=250" in seen[0]


async def test_activity_default_limit_and_no_optional_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _env({"data": []})
    seen = _route_cli(monkeypatch, {("audit", "log", "--limit=100"): envelope})
    async with _client() as client:
        resp = await client.get("/api/activity")
    assert resp.status_code == 200
    assert seen == [("audit", "log", "--limit=100")]


@pytest.mark.parametrize("limit", [0, 501, -1])
async def test_activity_limit_out_of_range_is_422(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    async def fail_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        raise AssertionError("run_cli must not be reached")

    monkeypatch.setattr(main_mod, "run_cli", fail_run_cli)
    async with _client() as client:
        resp = await client.get("/api/activity", params={"limit": limit})
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["code"] == "invalid_param"


@pytest.mark.parametrize(
    "params",
    [
        {"actor": "root"},  # not in {agent,human,system}
        {"action": "a b"},  # space outside the allowlist
        {"strategy": "-momo"},  # leading dash
        {"strategy": "mo/mo"},  # slash
    ],
)
async def test_activity_param_validation_is_422(
    monkeypatch: pytest.MonkeyPatch, params: dict[str, str]
) -> None:
    async def fail_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        raise AssertionError("run_cli must not be reached")

    monkeypatch.setattr(main_mod, "run_cli", fail_run_cli)
    async with _client() as client:
        resp = await client.get("/api/activity", params=params)
    assert resp.status_code == 422
    body = resp.json()
    assert body["ok"] is False
    assert body["code"] == "invalid_param"


# --- /api/ideas ---


async def test_ideas_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    ideas_data = {"data": [{"id": 1, "title": "vol carry"}]}  # bare array wrapped by the seam
    stats_data = {"window_days": 90, "counts": {"pool": 3}}
    _route_cli(
        monkeypatch,
        {
            ("research", "idea", "list"): _env(ideas_data, fetched_at="2026-08-09T00:00:05+00:00"),
            ("research", "idea", "stats"): _env(stats_data, fetched_at="2026-08-09T00:00:01+00:00"),
        },
    )
    async with _client() as client:
        resp = await client.get("/api/ideas")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["ideas"] == ideas_data
    assert body["stats"] == stats_data
    assert body["stats_window_days"] == 90
    assert body["fetched_at"] == "2026-08-09T00:00:01+00:00"  # min of the parts
    assert body["stale"] is False


async def test_ideas_stale_part_ors_into_composed_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_cli(
        monkeypatch,
        {
            ("research", "idea", "list"): _env({"data": []}, stale=True),
            ("research", "idea", "stats"): _env({"window_days": 90, "counts": {}}),
        },
    )
    async with _client() as client:
        resp = await client.get("/api/ideas")
    assert resp.status_code == 200
    assert resp.json()["stale"] is True


# --- /api/triage (Now screen: fleet + loops + capital) ---

_TRIAGE_FLEET = {"ok": True, "global_halt": False, "alerting": [],
                 "summary": {"total": 1, "alerting": 0, "by_health": {}},
                 "rows": [{"strategy": "a", "stage": "paper", "health": "ok"}]}
_TRIAGE_OPS = {"ok": False, "alerting": ["research"],
               "loops": {"research": {"health": "rate_limited", "detail": "quota"}}}
_TRIAGE_BOOK = {"ok": True, "capacity": 64, "allocated": 1,
                "unallocated_operational": [], "slices": []}


def _triage_routes(**overrides: Any) -> dict[tuple[str, ...], Any]:
    routes: dict[tuple[str, ...], Any] = {
        ("fleet", "health"): _env(_TRIAGE_FLEET),
        ("ops", "status"): _env(_TRIAGE_OPS),
        ("book", "status"): _env(_TRIAGE_BOOK),
    }
    routes.update({k: v for k, v in overrides.items()})  # type: ignore[misc]
    return routes


async def test_triage_ranks_across_all_three_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _route_cli(monkeypatch, _triage_routes())
    async with _client() as client:
        resp = await client.get("/api/triage")
    assert resp.status_code == 200
    body = resp.json()
    assert [item["kind"] for item in body["items"]] == ["loop_down"]
    assert body["sources"] == {"fleet": True, "ops": True, "book": True}
    assert body["headline"]["fleet_ok"] == 1


async def test_triage_degrades_per_part_instead_of_blanking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed source must be REPORTED, never silently rendered as all-clear."""
    routes = _triage_routes()
    routes[("ops", "status")] = CliError("cli_timeout", "ops status timed out")
    _route_cli(monkeypatch, routes)
    async with _client() as client:
        resp = await client.get("/api/triage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"]["ops"] is False
    assert body["items"] == []  # the ops-derived item is gone, but the screen still renders


async def test_triage_with_every_source_down_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering an empty 'all clear' from zero readable sources would be a lie."""
    err = CliError("cli_timeout", "timed out")
    _route_cli(monkeypatch, {("fleet", "health"): err, ("ops", "status"): err,
                             ("book", "status"): err})
    async with _client() as client:
        resp = await client.get("/api/triage")
    assert resp.status_code == 502
    assert resp.json()["code"] == "triage_unavailable"


async def test_ideas_cli_error_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    _route_cli(
        monkeypatch,
        {
            ("research", "idea", "list"): CliError("cli_timeout", "idea list timed out"),
            ("research", "idea", "stats"): _env({"window_days": 90, "counts": {}}),
        },
    )
    async with _client() as client:
        resp = await client.get("/api/ideas")
    assert resp.status_code == 502
    assert resp.json()["code"] == "cli_timeout"
