# FPE detection campaign report

## Headline

Across 420 injected faults in 2 models, 310 corrupted the output. Of those, 119 (38%) raised no IEEE exception at all — silent fixed-point corruption. Of the 191 that did raise an exception, 113 (59%) were masked before the output.

Detection over the 191 exception opportunities: flag polling 191/191 (100.0%), tensor scanning 188/191 (98.4%), output-only inspection 56/191 (29.3%).

## Table 1 — Fault outcome breakdown

| model | fault | n | rej | crash | inert | silent | exc-masked | exc+out |
|---|---|---|---|---|---|---|---|---|
| ssd_mobilenet_v2_coc… [tpu] | divzero | 30 | 0 | 0 | 29 | 1 | 0 | 0 |
| ssd_mobilenet_v2_coc… [tpu] | inf | 30 | 0 | 0 | 6 | 0 | 24 | 0 |
| ssd_mobilenet_v2_coc… [tpu] | invalid | 30 | 0 | 0 | 0 | 0 | 29 | 1 |
| ssd_mobilenet_v2_coc… [tpu] | overflow | 30 | 0 | 0 | 0 | 15 | 0 | 15 |
| ssd_mobilenet_v2_coc… [tpu] | scale_nan | 30 | 0 | 0 | 0 | 0 | 0 | 30 |
| ssd_mobilenet_v2_coc… [tpu] | underflow | 30 | 0 | 0 | 0 | 30 | 0 | 0 |
| ssd_mobilenet_v2_coc… [tpu] | zp_set | 30 | 0 | 0 | 1 | 29 | 0 | 0 |
| efficientdet_lite0_3… [tpu] | divzero | 30 | 0 | 0 | 29 | 1 | 0 | 0 |
| efficientdet_lite0_3… [tpu] | inf | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| efficientdet_lite0_3… [tpu] | invalid | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| efficientdet_lite0_3… [tpu] | overflow | 30 | 15 | 0 | 0 | 0 | 0 | 15 |
| efficientdet_lite0_3… [tpu] | scale_nan | 30 | 13 | 0 | 0 | 0 | 0 | 17 |
| efficientdet_lite0_3… [tpu] | underflow | 30 | 11 | 0 | 0 | 19 | 0 | 0 |
| efficientdet_lite0_3… [tpu] | zp_set | 30 | 0 | 0 | 6 | 24 | 0 | 0 |

## Table 2 — Detection rate by mechanism

Denominator: trials that raised an IEEE exception.

| model | n | flag polling | tensor scan | output only |
|---|---|---|---|---|
| ssd_mobilenet_v2_coc… [tpu] | 99 | 100.0% [96–100] | 97.0% [91–99] | 24.2% [17–34] |
| efficientdet_lite0_3… [tpu] | 92 | 100.0% [96–100] | 100.0% [96–100] | 34.8% [26–45] |

## Table 3 — IEEE 754 exception coverage

| exception | opportunities | detected | status |
|---|---|---|---|
| invalid | 187 | 187 | demonstrated, full detection |
| divbyzero | 0 | 0 | reachable in principle; not observed in this campaign |
| overflow | 27 | 27 | demonstrated, full detection |
| underflow | 30 | 30 | demonstrated, full detection |
| inexact | — | — | detectable but excluded by design (fires on ~all float ops) |

## Table 4 — Exception reachability across models

| model | trials | exception opportunities | silent corruption |
|---|---|---|---|
| ssd_mobilenet_v2_coc… [tpu] | 210 | 99 | 75 |
| efficientdet_lite0_3… [tpu] | 210 | 92 | 44 |
