# edgetpu-fpe

Runtime detection of IEEE 754 exceptions and int8 saturation during
quantized inference on the Coral Edge TPU.

## Install

Python 3.9 is required (no `pycoral` wheels above it) and NumPy must be
1.x (its C extension is built against NumPy 1).

```bash
conda create -n coral python=3.9 -y && conda activate coral
pip install --extra-index-url https://google-coral.github.io/py-repo/ \
    tflite-runtime==2.5.0.post1 pycoral==2.0.0
pip install "numpy<2" pillow tflite flatbuffers
```

```bash
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor | sudo tee /usr/share/keyrings/coral-edgetpu.gpg > /dev/null
sudo apt-get update && sudo apt-get install -y libedgetpu1-std edgetpu-compiler
```

If the device appears in `lsusb` but not in `list_edge_tpus()`:
`sudo usermod -aG plugdev $USER`, then log out and back in.

```bash
python scripts/check_env.py
bash scripts/fetch_models.sh
python scripts/make_inputs.py
```

## Run

```bash
python fpedetect/probe.py --model test_data/M.tflite --input inputs/X.npy --runs 5
python fpedetect/probe.py --model test_data/M_edgetpu.tflite --input inputs/X.npy --device tpu --runs 5
python fpedetect/characterize.py test_data/ --json results/surface.json
```

`--device tpu` requires a compiled `*_edgetpu.tflite`. `--no-preserve`
disables tensor preservation and is for overhead measurement only; with
it, post-hoc scanning reads recycled buffers.

## Reproduce

```bash
bash scripts/run_all.sh
```

Runs the CPU-path campaigns and writes `results/report_uniform/` and
`results/report_targeted/`. 15–25 min.

Device-side campaigns (targets are renumbered by compilation; list them
with `python eval/fault_inject.py list --model M_edgetpu.tflite --kind scale`):

```bash
R=results/campaigns
E=test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
EF=test_data/efficientdet_lite0_320_ptq_edgetpu.tflite

python eval/run_campaign.py --model $E --input inputs/ssd_input.npy \
    --device tpu --trials 30 --only 2,4 --json $R/ssd_targeted_tpu.json --csv $R/ssd_targeted_tpu.csv
python eval/run_campaign.py --model $EF --input inputs/effdet_input.npy \
    --device tpu --trials 30 --only 11,13 --json $R/effdet_targeted_tpu.json --csv $R/effdet_targeted_tpu.csv
python eval/make_report.py $R/*_tpu.json --out results/report_tpu
```

## Figures

```bash
python scripts/make_figures.py --outdir figs
```

## Notes

Injection sites are not fully seeded, so per-cell counts vary slightly
between runs. Cited values come from the committed files in
`results/campaigns/`.

Environment used: Ubuntu 24.04.1, Python 3.9.23, tflite-runtime
2.5.0.post1, pycoral 2.0.0, NumPy 1.26.4, Edge TPU Compiler
16.0.384591198, `libedgetpu1-std`, Coral USB Accelerator.