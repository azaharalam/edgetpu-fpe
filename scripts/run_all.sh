#!/usr/bin/env bash
# Reproduces every campaign behind the reported tables.
#
# Two sampling protocols, deliberately kept separate:
#   uniform  -- all scale sites sampled equally; measures how often an
#               exception is reachable in a model at all (Table 4)
#   targeted -- restricted to the dequantize inputs feeding post-processing;
#               measures detection where exceptions ARE reachable (Tables 2-3)
# Pooling them would confound reachability with sampling.
set -e
cd "$(dirname "$0")/.."
T=test_data; R=results/campaigns; mkdir -p "$R"

run() { python eval/run_campaign.py --model "$1" --input "$2" --trials 30 \
        ${4:+--only "$4"} --json "$R/$3.json" --csv "$R/$3.csv"; }

run $T/ssd_mobilenet_v2_coco_quant_postprocess.tflite inputs/ssd_input.npy      ssd_uniform
run $T/efficientdet_lite0_320_ptq.tflite              inputs/effdet_input.npy   effdet_uniform
run $T/mobilenet_v2_1.0_224_quant.tflite              inputs/mobilenet_input.npy mobilenet_uniform
run $T/movenet_single_pose_lightning_ptq.tflite       inputs/movenet_person.npy movenet_uniform
run $T/deeplabv3_mnv2_pascal_quant.tflite             ""                        deeplab_uniform

run $T/ssd_mobilenet_v2_coco_quant_postprocess.tflite inputs/ssd_input.npy    ssd_targeted    250,257
run $T/efficientdet_lite0_320_ptq.tflite              inputs/effdet_input.npy effdet_targeted 593,596

python eval/make_report.py $R/*_uniform.json  --out results/report_uniform
python eval/make_report.py $R/*_targeted.json --out results/report_targeted
