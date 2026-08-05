# FPE detection campaign report

## Headline

Across 450 injected faults in 2 models, 319 corrupted the output. Of those, 133 (42%) raised no IEEE exception at all — silent fixed-point corruption. Of the 186 that did raise an exception, 108 (58%) were masked before the output.

Detection over the 186 exception opportunities: flag polling 186/186 (100.0%), tensor scanning 182/186 (97.8%), output-only inspection 51/186 (27.4%).

## Table 1 — Fault outcome breakdown

| model | fault | n | rej | crash | inert | silent | exc-masked | exc+out |
|---|---|---|---|---|---|---|---|---|
| ssd_mobilenet_v2_coco_quan [ | divzero | 30 | 0 | 0 | 30 | 0 | 0 | 0 |
| ssd_mobilenet_v2_coco_quan [ | inf | 30 | 0 | 0 | 2 | 0 | 26 | 2 |
| ssd_mobilenet_v2_coco_quan [ | invalid | 30 | 0 | 0 | 0 | 0 | 29 | 1 |
| ssd_mobilenet_v2_coco_quan [ | overflow | 30 | 16 | 0 | 0 | 0 | 0 | 14 |
| ssd_mobilenet_v2_coco_quan [ | saturate | 30 | 0 | 0 | 9 | 21 | 0 | 0 |
| ssd_mobilenet_v2_coco_quan [ | scale_nan | 30 | 15 | 0 | 0 | 0 | 0 | 15 |
| ssd_mobilenet_v2_coco_quan [ | underflow | 30 | 19 | 0 | 0 | 11 | 0 | 0 |
| ssd_mobilenet_v2_coco_quan [ | zp_set | 30 | 0 | 0 | 4 | 26 | 0 | 0 |
| ssd_mobilenet_v2_coco_quan [ | divzero | 30 | 0 | 0 | 29 | 1 | 0 | 0 |
| ssd_mobilenet_v2_coco_quan [ | inf | 30 | 0 | 0 | 6 | 0 | 24 | 0 |
| ssd_mobilenet_v2_coco_quan [ | invalid | 30 | 0 | 0 | 0 | 0 | 29 | 1 |
| ssd_mobilenet_v2_coco_quan [ | overflow | 30 | 0 | 0 | 0 | 15 | 0 | 15 |
| ssd_mobilenet_v2_coco_quan [ | scale_nan | 30 | 0 | 0 | 0 | 0 | 0 | 30 |
| ssd_mobilenet_v2_coco_quan [ | underflow | 30 | 0 | 0 | 0 | 30 | 0 | 0 |
| ssd_mobilenet_v2_coco_quan [ | zp_set | 30 | 0 | 0 | 1 | 29 | 0 | 0 |

## Table 2 — Detection rate by mechanism

Denominator: trials that raised an IEEE exception.

| model | n | flag polling | tensor scan | output only |
|---|---|---|---|---|
| ssd_mobilenet_v2_coco_quan [ | 87 | 100.0% [96–100] | 98.9% [94–100] | 31.0% [22–41] |
| ssd_mobilenet_v2_coco_quan [ | 99 | 100.0% [96–100] | 97.0% [91–99] | 24.2% [17–34] |

## Table 3 — IEEE 754 exception coverage

| exception | opportunities | detected | status |
|---|---|---|---|
| invalid | 180 | 180 | demonstrated, full detection |
| divbyzero | 0 | 0 | reachable in principle; not observed in this campaign |
| overflow | 25 | 25 | demonstrated, full detection |
| underflow | 29 | 29 | demonstrated, full detection |
| inexact | — | — | detectable but excluded by design (fires on ~all float ops) |

## Table 4 — Exception reachability across models

| model | trials | exception opportunities | silent corruption |
|---|---|---|---|
| ssd_mobilenet_v2_coco_quan [cpu] | 240 | 87 | 58 |
| ssd_mobilenet_v2_coco_quan [tpu] | 210 | 99 | 75 |
