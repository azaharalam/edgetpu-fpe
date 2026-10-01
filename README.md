# edgetpu-fpe

Runtime detection of IEEE 754 exceptions and int8 saturation during
quantized inference on the Coral Edge TPU. Artifact for the SC26 research
poster *Reliable Numeric Fault Detection During Edge TPU Inference*.

Three mechanisms, none of which modify the compiler, runtime, or model:

| mechanism | what it reads | output |
|---|---|---|
| status-register polling | host FP status register (`feclearexcept`/`fetestexcept` around `Invoke()`) | one sticky bit per exception class |
| tensor scanning | every `float32` tensor in the arena after `Invoke()` | NaN / ±Inf / subnormal counts per tensor |
| saturation counting | every `int8`/`uint8` activation | elements at a clamp bound that is not the zero point |

```
fpedetect/probe.py          the detector (all three mechanisms, CPU or TPU path)
fpedetect/characterize.py   float32 / int8 surface of each model, partition split
eval/fault_inject.py        size-preserving fault injection into a .tflite file
eval/run_campaign.py        injection campaign, one child process per trial
eval/make_report.py         campaign JSON -> markdown tables (report.md)
scripts/                    setup, reproduction, figures
results/campaigns/          the ten campaigns behind every reported number
results/surface.json        output of characterize.py on the five models
```

## 1. Install

Python 3.9 is required (no `pycoral` wheels above it) and NumPy must be
1.x (its C extension is built against NumPy 1). Everything below runs
without an Edge TPU; the device is needed only for the `--device tpu`
commands in §4.

```bash
conda create -n coral python=3.9 -y && conda activate coral
pip install --extra-index-url https://google-coral.github.io/py-repo/ \
    tflite-runtime==2.5.0.post1 pycoral==2.0.0
pip install "numpy<2" pillow tflite flatbuffers matplotlib
```

Edge TPU runtime and compiler (Debian/Ubuntu):

```bash
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor | sudo tee /usr/share/keyrings/coral-edgetpu.gpg > /dev/null
sudo apt-get update && sudo apt-get install -y libedgetpu1-std edgetpu-compiler
```

If the USB Accelerator appears in `lsusb` but not in
`edgetpu.list_edge_tpus()`: `sudo usermod -aG plugdev $USER`, then log out
and back in.

Models and inputs (models come from Google's `test_data` repository and
are not vendored; `*_edgetpu.tflite` files there are already compiled):

```bash
python scripts/check_env.py        # fails loudly on the wrong Python/NumPy
bash   scripts/fetch_models.sh     # clones google-coral/test_data
python scripts/make_inputs.py      # writes inputs/*.npy at each model's resolution
```

## 2. Test

A two-minute smoke test on the CPU path: environment check, one probed
inference, one verified injection, a one-trial-per-class campaign, its
report, and the summary and poster figures.

```bash
bash scripts/smoke_test.sh
```

It ends with `smoke test passed`. Figures land in `figs/`, the report in
`results/report_smoke/report.md`.

## 3. Run the detector

```bash
# CPU path
python fpedetect/probe.py --model test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite \
    --input inputs/ssd_input.npy --runs 5
# accelerator path
python fpedetect/probe.py --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
    --input inputs/ssd_input.npy --device tpu --runs 5
# float32 / int8 surface and TPU-CPU partition of every model in a directory
python fpedetect/characterize.py test_data/ --json results/surface.json
```

`probe.py` prints the exception flags raised, the tensors holding
NaN/Inf/subnormals, the saturation count, and median latency over the
runs. `--json FILE` writes the same per run. `--no-preserve` turns off
`preserve_all_tensors` and is for overhead measurement only; with it the
scan reads recycled buffers and reports false NaNs. `--device tpu`
requires a compiled `*_edgetpu.tflite` and an attached accelerator.
`characterize.py --skip-compiler` omits the partition split when
`edgetpu_compiler` is not installed.

Inject a single fault by hand:

```bash
python eval/fault_inject.py list --model test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite --kind scale
python eval/fault_inject.py inject --model test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite \
    --out /tmp/faulty.tflite --target 250 --fault scale_nan --verify --input inputs/ssd_input.npy
```

Fault types: `invalid`, `inf`, `divzero` (float constants); `saturate`
(integer constants); `overflow`, `underflow`, `scale_nan`, `scale_zero`,
`scale_negative`, `zp_set` (quantization metadata). `--verify` reruns the
patched model and reports the outcome class.

## 4. Reproduce the campaigns

Two sampling protocols, kept separate because pooling them would confound
reachability with sampling: **uniform** draws injection sites from all
metadata of a model; **targeted** restricts scale and zero-point faults to
the dequantization inputs feeding post-processing. `--trials N` means N
trials per fault class.

CPU path, all five models, 15–25 min (930 uniform + 480 targeted trials):

```bash
bash scripts/run_all.sh
```

writes `results/campaigns/*.json|csv`, `results/report_uniform/` and
`results/report_targeted/`.

Accelerator path (compilation renumbers tensors, so the targeted indices
differ from the CPU ones; list them with `fault_inject.py list --kind scale`):

```bash
R=results/campaigns
E=test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
EF=test_data/efficientdet_lite0_320_ptq_edgetpu.tflite

python eval/run_campaign.py --model $E  --input inputs/ssd_input.npy    --device tpu --trials 30 \
    --json $R/ssd_uniform_tpu.json --csv $R/ssd_uniform_tpu.csv
python eval/run_campaign.py --model $E  --input inputs/ssd_input.npy    --device tpu --trials 30 --only 2,4 \
    --json $R/ssd_targeted_tpu.json --csv $R/ssd_targeted_tpu.csv
python eval/run_campaign.py --model $EF --input inputs/effdet_input.npy --device tpu --trials 30 --only 11,13 \
    --json $R/effdet_targeted_tpu.json --csv $R/effdet_targeted_tpu.csv
python eval/make_report.py $R/*_targeted*.json --out results/report_targeted_both
```

The committed campaigns total 2,040 trials: 930 uniform CPU, 480
targeted CPU, 420 targeted TPU, 210 uniform TPU. The reported detection
rates (338 of 338 polling, 334 of 338 scanning, 83 of 338 output
inspection) come from `make_report.py` over the four targeted campaigns.

## 5. Figures

All figure scripts carry their data as tables transcribed from
`results/report_*/report.md` and the `probe.py` latency output;
re-running a campaign means updating those tables, not the plotting code. Every script takes `--outdir` and
`--only NAME`.

Summary (IEEEtran, one column wide, Times metrics, 8 pt):

```bash
python scripts/make_paper_figures.py --outdir figures
#   fig_overhead_detection  Fig. 2   fig_reachability  Fig. 3   fig_delegation  Fig. 4
```

Poster (six evaluation panels, 6.3 × 5.0 in each; text is drawn as
outlines in the SVGs, so no font is needed to view them; Carlito or
Calibri must be installed to render them):

```bash
python scripts/make_poster_figure.py --outdir figs
#   p1_overhead  p2_detection  p3_reachability   Fig. 2
#   p4_outcomes  p5_visibility p6_validation     Fig. 3
```

`sudo apt-get install fonts-crosextra-carlito && rm -rf ~/.cache/matplotlib`
installs Calibri-metric fonts on Linux. `scripts/make_figures.py` is the
earlier full-width (7.16 in) version of the summary figures and is kept
for reference.

## Notes

Injection sites are not fully seeded, so per-cell counts vary slightly
between runs. Cited values come from the committed files in `results/campaigns/`.

Environment used: Ubuntu 24.04.1, Python 3.9.23, tflite-runtime
2.5.0.post1, pycoral 2.0.0, NumPy 1.26.4, Edge TPU
Compiler 16.0.384591198, `libedgetpu1-std`, Coral USB Accelerator.
`environment.yml` and `requirements-frozen.txt` pin the exact stack.