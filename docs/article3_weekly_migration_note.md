# Article 3 weekly migration note

Article 3 specification version 2.0 replaces the earlier Article-3 handover. It does not recreate the historical monthly pipeline.

The implementation consumes the existing verified `daily_proxy_2011` weekly snapshot, whose data contract is weekly Friday (`W-FRI`) EUR decimal log returns with `52` steps per year. Therefore a five-year horizon is `260` weekly steps and terminal horizons are `52`, `156`, and `260` steps. Monthly-only helpers remain available for legacy consumers, but weekly calls must use the explicit step-based API.

The weekly reference is a synchronized MBB: a single source time index drives all 12 weekly return series in a scenario. This preserves contemporaneous cross-driver vectors. The independent-IID control intentionally samples each driver independently and is used only to identify the effect of that synchronization.

All data provenance is validated against the snapshot manifest and profile before a run begins. A mismatch in profile, frequency, date range, dimensions, driver registry, or file hash stops the run. No live download, source substitution, or unmarked toy data is permitted in the full research workflow.
