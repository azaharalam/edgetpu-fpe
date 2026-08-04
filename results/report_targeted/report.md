# FPE detection campaign report

## Headline

Across 480 injected faults in 2 models, 239 corrupted the output. Of those, 92 (38%) raised no IEEE exception at all — silent fixed-point corruption. Of the 147 that did raise an exception, 115 (78%) were masked before the output.

Detection over the 147 exception opportunities: flag polling 147/147 (100.0%), tensor scanning 146/147 (99.3%), output-only inspection 27/147 (18.4%).

## Table 1 — Fault outcome breakdown

| model | fault | n | rej | crash | inert | silent | exc-masked | exc+out |
|---|---|---|---|---|---|---|---|---|
| efficientdet_lite0_320_ptq.t | divzero | 30 | 0 | 0 | 30 | 0 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | inf | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| efficientdet_lite0_320_ptq.t | invalid | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| efficientdet_lite0_320_ptq.t | overflow | 30 | 30 | 0 | 0 | 0 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | saturate | 30 | 0 | 0 | 5 | 25 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | scale_nan | 30 | 30 | 0 | 0 | 0 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | underflow | 30 | 30 | 0 | 0 | 0 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | zp_set | 30 | 14 | 0 | 7 | 9 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | divzero | 30 | 0 | 0 | 30 | 0 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | inf | 30 | 0 | 0 | 2 | 0 | 26 | 2 |
| ssd_mobilenet_v2_coco_quant_ | invalid | 30 | 0 | 0 | 0 | 0 | 29 | 1 |
| ssd_mobilenet_v2_coco_quant_ | overflow | 30 | 16 | 0 | 0 | 0 | 0 | 14 |
| ssd_mobilenet_v2_coco_quant_ | saturate | 30 | 0 | 0 | 9 | 21 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | scale_nan | 30 | 15 | 0 | 0 | 0 | 0 | 15 |
| ssd_mobilenet_v2_coco_quant_ | underflow | 30 | 19 | 0 | 0 | 11 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | zp_set | 30 | 0 | 0 | 4 | 26 | 0 | 0 |

## Table 2 — Detection rate by mechanism

Denominator: trials that raised an IEEE exception.

| model | n | flag polling | tensor scan | output only |
|---|---|---|---|---|
| efficientdet_lite0_320_ptq.t | 60 | 100.0% [94–100] | 100.0% [94–100] | 0.0% [0–6] |
| ssd_mobilenet_v2_coco_quant_ | 87 | 100.0% [96–100] | 98.9% [94–100] | 31.0% [22–41] |

## Table 3 — IEEE 754 exception coverage

| exception | opportunities | detected | status |
|---|---|---|---|
| invalid | 145 | 145 | demonstrated, full detection |
| divbyzero | 0 | 0 | reachable in principle; not observed in this campaign |
| overflow | 13 | 13 | demonstrated, full detection |
| underflow | 14 | 14 | demonstrated, full detection |
| inexact | — | — | detectable but excluded by design (fires on ~all float ops) |

## Table 4 — Exception reachability across models

| model | trials | exception opportunities | silent corruption |
|---|---|---|---|
| efficientdet_lite0_320_ptq.tflite | 240 | 60 | 34 |
| ssd_mobilenet_v2_coco_quant_postpr | 240 | 87 | 58 |
