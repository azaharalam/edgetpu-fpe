#!/usr/bin/env bash
# Smoke test on the CPU path, about two minutes. Needs §1 of the README done
# (environment, test_data/, inputs/). No Edge TPU required.
set -e
cd "$(dirname "$0")/.."
M=test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite
X=inputs/ssd_input.npy
[ -f "$M" ] || { echo "run: bash scripts/fetch_models.sh"; exit 1; }
[ -f "$X" ] || { echo "run: python scripts/make_inputs.py"; exit 1; }

echo "== 1/5 environment";  python scripts/check_env.py
echo "== 2/5 probe";        python fpedetect/probe.py --model $M --input $X --runs 2
echo "== 3/5 injection";    python eval/fault_inject.py inject --model $M --out /tmp/smoke_faulty.tflite \
                                --target 250 --fault scale_nan --verify --input $X
echo "== 4/5 campaign";     python eval/run_campaign.py --model $M --input $X --trials 1 --only 250,257 \
                                --json /tmp/smoke_campaign.json --csv /tmp/smoke_campaign.csv
                            python eval/make_report.py /tmp/smoke_campaign.json --out results/report_smoke
echo "== 5/5 figures"
for s in make_paper_figures make_poster_figure; do
    python scripts/$s.py --outdir figs
done
echo; echo "smoke test passed"
