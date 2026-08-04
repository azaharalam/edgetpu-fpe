# FPE detection campaign report

## Headline

Across 930 injected faults in 5 models, 424 corrupted the output. Of those, 304 (72%) raised no IEEE exception at all — silent fixed-point corruption. Of the 120 that did raise an exception, 115 (96%) were masked before the output.

Detection over the 120 exception opportunities: flag polling 120/120 (100.0%), tensor scanning 118/120 (98.3%), output-only inspection 3/120 (2.5%).

## Table 1 — Fault outcome breakdown

| model | fault | n | rej | crash | inert | silent | exc-masked | exc+out |
|---|---|---|---|---|---|---|---|---|
| deeplabv3_mnv2_pascal_quant. | overflow | 30 | 27 | 0 | 0 | 3 | 0 | 0 |
| deeplabv3_mnv2_pascal_quant. | saturate | 30 | 0 | 0 | 3 | 27 | 0 | 0 |
| deeplabv3_mnv2_pascal_quant. | scale_nan | 30 | 28 | 1 | 1 | 0 | 0 | 0 |
| deeplabv3_mnv2_pascal_quant. | underflow | 30 | 17 | 0 | 2 | 11 | 0 | 0 |
| deeplabv3_mnv2_pascal_quant. | zp_set | 30 | 8 | 0 | 5 | 17 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | divzero | 30 | 0 | 0 | 30 | 0 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | inf | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| efficientdet_lite0_320_ptq.t | invalid | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| efficientdet_lite0_320_ptq.t | overflow | 30 | 0 | 0 | 11 | 19 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | saturate | 30 | 0 | 0 | 5 | 25 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | scale_nan | 30 | 1 | 6 | 8 | 15 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | underflow | 30 | 1 | 2 | 9 | 18 | 0 | 0 |
| efficientdet_lite0_320_ptq.t | zp_set | 30 | 0 | 0 | 21 | 9 | 0 | 0 |
| mobilenet_v2_1.0_224_quant.t | overflow | 30 | 27 | 0 | 2 | 1 | 0 | 0 |
| mobilenet_v2_1.0_224_quant.t | saturate | 30 | 0 | 0 | 29 | 1 | 0 | 0 |
| mobilenet_v2_1.0_224_quant.t | scale_nan | 30 | 27 | 2 | 0 | 0 | 0 | 1 |
| mobilenet_v2_1.0_224_quant.t | underflow | 30 | 17 | 1 | 3 | 9 | 0 | 0 |
| mobilenet_v2_1.0_224_quant.t | zp_set | 30 | 8 | 0 | 5 | 17 | 0 | 0 |
| movenet_single_pose_lightnin | overflow | 30 | 0 | 0 | 9 | 21 | 0 | 0 |
| movenet_single_pose_lightnin | saturate | 30 | 0 | 0 | 13 | 17 | 0 | 0 |
| movenet_single_pose_lightnin | scale_nan | 30 | 2 | 8 | 9 | 11 | 0 | 0 |
| movenet_single_pose_lightnin | underflow | 30 | 4 | 1 | 9 | 16 | 0 | 0 |
| movenet_single_pose_lightnin | zp_set | 30 | 2 | 0 | 23 | 5 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | divzero | 30 | 0 | 0 | 30 | 0 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | inf | 30 | 0 | 0 | 2 | 0 | 26 | 2 |
| ssd_mobilenet_v2_coco_quant_ | invalid | 30 | 0 | 0 | 0 | 0 | 29 | 1 |
| ssd_mobilenet_v2_coco_quant_ | overflow | 30 | 29 | 0 | 0 | 1 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | saturate | 30 | 0 | 0 | 9 | 21 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | scale_nan | 30 | 25 | 2 | 0 | 2 | 0 | 1 |
| ssd_mobilenet_v2_coco_quant_ | underflow | 30 | 11 | 0 | 1 | 18 | 0 | 0 |
| ssd_mobilenet_v2_coco_quant_ | zp_set | 30 | 8 | 0 | 2 | 20 | 0 | 0 |

## Table 2 — Detection rate by mechanism

Denominator: trials that raised an IEEE exception.

| model | n | flag polling | tensor scan | output only |
|---|---|---|---|---|
| deeplabv3_mnv2_pascal_quant. | 0 | — | — | — |
| efficientdet_lite0_320_ptq.t | 60 | 100.0% [94–100] | 100.0% [94–100] | 0.0% [0–6] |
| mobilenet_v2_1.0_224_quant.t | 1 | 100.0% [21–100] | 0.0% [0–79] | 0.0% [0–79] |
| movenet_single_pose_lightnin | 0 | — | — | — |
| ssd_mobilenet_v2_coco_quant_ | 59 | 100.0% [94–100] | 98.3% [91–100] | 5.1% [2–14] |

## Table 3 — IEEE 754 exception coverage

| exception | opportunities | detected | status |
|---|---|---|---|
| invalid | 120 | 120 | demonstrated, full detection |
| divbyzero | 0 | 0 | reachable in principle; not observed in this campaign |
| overflow | 0 | 0 | reachable in principle; not observed in this campaign |
| underflow | 0 | 0 | reachable in principle; not observed in this campaign |
| inexact | — | — | detectable but excluded by design (fires on ~all float ops) |

## Table 4 — Exception reachability across models

| model | trials | exception opportunities | silent corruption |
|---|---|---|---|
| deeplabv3_mnv2_pascal_quant.tflite | 150 | 0 | 58 |
| efficientdet_lite0_320_ptq.tflite | 240 | 60 | 86 |
| mobilenet_v2_1.0_224_quant.tflite | 150 | 1 | 28 |
| movenet_single_pose_lightning_ptq. | 150 | 0 | 70 |
| ssd_mobilenet_v2_coco_quant_postpr | 240 | 59 | 62 |
