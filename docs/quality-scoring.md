# Quality scoring

Per the platform design, each session gets a weighted 0-100 score:

| Component        | Weight | Source in spatial-os |
|-------------------|--------|-----------------------|
| Coverage           | 30%    | **Estimated**: top-down XY grid occupancy of the reconstructed cloud. Override with `coverage_pct` in `session_metadata.json` if you have the mobile app's real live coverage %. |
| Depth Quality       | 25%    | **Real**: derived from `device.depth_source` (`lidar` > `tof` > `ml_estimated`), as declared in the session manifest. |
| Trajectory          | 25%    | **Estimated**: path length + start/end proximity (loop-closure proxy) computed from the pose stream. Override `has_loop_closure` in `session_metadata.json` for a real SLAM signal. |
| Density             | 20%    | **Estimated**: points/m² of the reconstructed cloud, penalized by outlier ratio. |

## Why "estimated"?

The design doc's Coverage and Trajectory scores assume live telemetry from
the mobile scanning app (a real-time coverage overlay, SLAM loop-closure
detection) that doesn't exist for a backend processing already-captured
files. `spatial-os` computes a best-effort estimate from the reconstructed
geometry and pose stream instead, and marks each component
`"estimated": true`/`false` in the output JSON so this limitation is visible
rather than silently presented as ground truth.

If the mobile app (or any other upstream source) can supply the real values,
drop a `session_metadata.json` next to a session's `manifest.json`:

```json
{
  "coverage_pct": 82.5,
  "has_loop_closure": true,
  "depth_source": "lidar"
}
```

Any field present there takes priority over the derived estimate.

## Thresholds

- `< 40` → reject, session should be re-scanned.
- `40-70` → accept, but flagged low quality (not suitable for robotics
  training per the design doc).
- `> 70` → accept fully.
