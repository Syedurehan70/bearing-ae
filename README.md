# Unsupervised bearing fault detection under operating-condition shift

An autoencoder trained **only on healthy vibration data from a single motor load**,
used to flag rolling-element bearing faults by reconstruction error, on the CWRU
12 kHz drive-end dataset.

Detecting seeded faults on CWRU is not the interesting part — the classes are
close to linearly separable and a scalar kurtosis threshold already gets most of
the way there. This repository is set up to measure the part that actually
decides whether an unsupervised detector is deployable: **does the decision
threshold survive a change in operating condition it was never calibrated on?**

## Protocol

1. Split the healthy windows recorded at **one** motor load (1 hp by default)
   three ways: train, calibrate, test. No faulty window is seen during training.
2. Fix the anomaly threshold at the 99th percentile of reconstruction error on
   the **calibration** split. Fault labels are never used to select it. The
   healthy **test** split is touched by neither training nor calibration, so the
   false-alarm rate at the training load is an independent measurement rather
   than a restatement of the quantile that defined the threshold.
3. Score faults at the training load — the easy, usually-reported case.
4. Score healthy **and** faulty data at the three unseen loads (0, 2, 3 hp), and
   report the false-alarm rate on healthy data at each load separately.
5. Run the same protocol with a spectral-kurtosis-style scalar detector, so the
   autoencoder has an honest baseline to beat rather than being reported alone.

Labels enter the pipeline only in steps 3–5, for scoring.

## Results

> Fill this table in from `results/metrics.json` after your run, and delete this
> line. Do not commit the repository with placeholders in it.

| | training load (0 hp) | unseen loads (1–3 hp) |
|---|---|---|
| ROC-AUC (dense AE, log-spectrum) | — | — |
| True positive rate @ fixed threshold | — | — |
| **False positive rate on healthy data** | — | — |
| ROC-AUC (kurtosis baseline) | — | — |
| False positive rate (kurtosis baseline) | — | — |

False-alarm rate on healthy data, by motor load:

![false positives by load](results/fpr_by_load.png)

The expected and important observation is that the false-positive rate rises
once the operating condition moves away from the one the threshold was
calibrated on. The autoencoder responds to *any* departure from the training
distribution, and a load or speed change is such a departure. Reconstruction
error is a novelty score, not a damage score — which is precisely why deployed
condition monitoring needs either per-regime thresholds, operating-condition
conditioning, or features that are invariant to load.

## Running it

```bash
pip install -r requirements.txt
python -m src.main --config configs/dense_logspec.yaml --out results
python -m src.main --config configs/conv_raw.yaml --out results_conv
```

Point `data.roots` in the config at a directory containing the CWRU `.mat`
files. The loader walks it recursively and handles both the original numeric
file names (`97.mat`, `105.mat`) and the descriptive ones used by most public
mirrors (`Normal_0.mat`, `IR007_1.mat`, `OR007@6_2.mat`). Set `data.exclude` to
keep a single sampling rate — several mirrors ship the 48 kHz drive-end
recordings next to the 12 kHz ones, and mixing sampling rates silently corrupts
every spectral feature.

Everything that affects a result lives in the YAML config, including the seed.
`notebooks/kaggle_run.ipynb` runs the same code on Kaggle against a mounted CWRU
dataset.

## Design choices worth arguing with

- **Healthy and faulty recordings are brought to a common sampling rate first.**
  CWRU records the healthy baseline at 48 kHz and the drive-end faults at 12 kHz,
  and stores the rate nowhere in the files. Left alone, the autoencoder separates
  the two classes on sampling rate rather than on damage, and reports a near
  perfect score for the wrong reason. Everything is resampled to 12 kHz before
  any feature is computed, and the resolved rate and resulting duration are
  printed for every run so the assumption can be checked rather than trusted.
- **Windows are RMS-normalised before featurisation.** Without this, the model
  can separate healthy from faulty on overall vibration energy alone, and the
  reported numbers say nothing about whether the autoencoder learned structure.
- **Training uses the 1 hp baseline, not 0 hp.** The 0 hp healthy recording is
  roughly five seconds long. Fitting an autoencoder to the few dozen windows it
  yields produces a model that has memorised its training set, and every
  subsequent number is an artifact of that rather than a property of the method.
- **The bottleneck is small (8 units).** A wide bottleneck approaches the
  identity function and reconstructs faults as accurately as healthy data,
  which quietly destroys the detector.
- **The threshold is a quantile of healthy validation error**, not the value
  that maximises accuracy on the test set. Tuning it against labels would make
  the method supervised while still calling itself unsupervised — a common
  failure in published CWRU results.

## What this does not show

- CWRU faults are **seeded by electro-discharge machining**, not grown in
  service. They are more localised and more visible than real spalling. Nothing
  here demonstrates early-stage or progressive damage detection; NASA's IMS
  run-to-failure set is the right follow-up for that.
- One test rig, one bearing type, one fault-injection method. No claim of
  cross-machine generalisation.
- No physics is used. Bearing characteristic frequencies (BPFO/BPFI/BSF) are
  computable from geometry and shaft speed here, and an envelope-spectrum
  detector built on them is the correct engineering baseline. It is not
  implemented in this repository.
- Healthy data is scarce. A single CWRU baseline recording is 5-10 s, which
  after windowing leaves a few hundred training windows with heavy overlap
  between them. They are not independent samples, and the model remains
  overparameterised relative to them.
- Two-hour project. The point was a clean protocol and an honest failure mode,
  not a state-of-the-art number.

## Data

Case Western Reserve University Bearing Data Center, 12 kHz drive-end fault data
and normal baseline data: <https://engineering.case.edu/bearingdatacenter>

## Licence

MIT.
