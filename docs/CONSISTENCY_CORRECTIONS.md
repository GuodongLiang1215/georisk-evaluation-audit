# Corrections from the reviewer consistency audit

The strict binary comparison `coverage <= 0.5` omitted the nominal endpoint stored as `0.5000000000000001`. The binary analysis now uses `coverage <= 0.5 + 1e-12`, matching the inclusive 0.10–0.50 definition. The original coverage grid is preserved; no selection arithmetic or scores were changed.

| Endpoint | P1 earlier 0.10–0.45 | P1 inclusive 0.10–0.50 | P2 earlier 0.10–0.45 | P2 inclusive 0.10–0.50 |
|---|---:|---:|---:|---:|
| Pooled mIoU | 55.9% | 62.1% | 60.3% | 67.8% |
| Foreground Dice | 27.8% | 34.0% | 47.2% | 56.7% |

`gain_at_coverage_le_0_5` and `fraction_gain_at_coverage_le_0_5` are the only changed columns in `coverage_gain_decomposition_endpoints.csv`. The corrected reference was separately reconstructed by summing the eight adjacent trapezoids from 0.10 through 0.50. Full AURCs, complete curves and all tile selections are unchanged. The multiclass interval correction remains 83.3%, and the overall multiclass displacement remains 32.7%.

Section 3.2 now follows the plotted source curves at coverage 0.20 (−0.0654/−0.0755 pooled-mIoU differences in P1/P2) and rounds the P1 foreground-Dice difference at 0.80 to −0.0188. These were text corrections; curve values did not change.

The supplementary reproduction guide now describes the actual package paths and distinguishes computed outputs, table snapshots, retained supplementary figures and optional model inference. The comparator and temperature protocols are included as `COMPARATOR_PROTOCOL.md` and `TEMPERATURE_PROTOCOL.md`.
