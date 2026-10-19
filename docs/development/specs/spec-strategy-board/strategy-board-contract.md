# Strategy board response contract

`algua runs board` is a pure read. The monitor exposes the same data at
`GET /api/strategy-board` through the existing subprocess JSON seam.

```json
{
  "schema_version": "strategy-board/v1",
  "count": 1,
  "rows": [
    {
      "strategy_id": 22,
      "name": "high52_proximity",
      "family": null,
      "stage": "paper",
      "latest_gate": {
        "run_id": 901,
        "created_at": "2026-10-15T12:00:00+00:00",
        "passed": true,
        "sharpe_oos": 0.81,
        "total_return_oos": 0.12,
        "max_drawdown_oos": -0.08,
        "n_obs_oos": 126,
        "mean_window_sharpe": 0.73,
        "min_window_sharpe": 0.19,
        "pct_positive_windows": 0.75
      }
    }
  ]
}
```

## Field rules

- `schema_version` is exactly `strategy-board/v1`.
- `count` equals `rows.length` and counts registered strategy identities, not runs.
- `rows` follow lifecycle order, then case-insensitive name, then `strategy_id`.
- `strategy_id`, `name`, `family`, and `stage` come from the registry row.
- `latest_gate` is nullable. It is selected only from `runs.kind = 'gate'` where
  `runs.strategy_id = strategies.id`, ordered by `created_at DESC, id DESC`. If that selected
  row has no valid offset-aware ISO timestamp, its recency cannot be trusted and `latest_gate`
  is `null` rather than falling back to a more favorable older run.
- Every metric is a nullable finite scalar already present in the fixed run vocabulary. No metric
  is derived, imputed, or normalized by this projection.
- The contract never returns `derived_from`, config, components, hashes, gate decisions, return
  series, or holdout blobs.

Client-side stage filtering and exploratory ordering operate only over this complete payload. They
may reorder rows but may never replace `latest_gate` or feed a promotion path.
