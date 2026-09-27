# Strategy Specification

Frozen selection: `NO_TRADE`. Frozen diagnostic controller quotes a side only when fill probability × (side-specific predicted markout in ticks − registered cost in ticks) is positive and adverse probability is below 0.7. AS/GLFT approximations remain honestly named `AS_LIKE_HEURISTIC` and `GLFT_LIKE_HEURISTIC`; faithful published calibration was not identified by the limited public samples.
