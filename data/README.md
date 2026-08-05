# Datasets

Four publicly available EEG datasets are used across the MV-AFA paper and its benchmarks.
None of the raw data is included in this repository — download instructions are provided below.

## Labeled samples used in the analysis

The table below reports the labeled sample counts actually used in our experiments, taken
from the experimental metadata **after dataset-specific channel adaptation**. TUSZ contains
mixed sampling rates in this subset: 250, 256, 400, 512, and 1000 Hz.

| Dataset | Subjects | Files | Total samples | Positive | Negative | Rate (Hz) | Channels |
|---------|----------|-------|---------------|----------|----------|-----------|----------|
| CHB-MIT | 24 | 673 EDF | 3,493,603 | 10,927 | 3,482,676 | 256 | 18 bipolar |
| Siena | 14 | 41 EDF | 9,344 | 1,153 | 8,191 | 512 | 16 bipolar |
| SeizeIT2 | 125 | 2,846 EDF | 578,390 | 13,074 | 565,316 | 256 | 2 EEG |
| TUSZ | 130 | 1,237 EDF | 149,133 | 19,080 | 130,053 | 250–1000 | 20 TCP bipolar |

---

## 1. CHB-MIT Scalp EEG Database

**Used by:** All four benchmark methods + MV-AFA

### Description

The CHB-MIT Scalp EEG Database was collected at Boston Children's Hospital and is jointly
maintained by MIT. It contains long-term, continuous scalp EEG recordings from **24 paediatric
patients** (ages 1.5–22 years) with intractable epilepsy. Recordings span multiple sessions
per patient and cover both ictal (seizure) and interictal (non-seizure) periods.

| Property | Value |
|----------|-------|
| Subjects | 24 (paediatric) |
| Files used | 673 EDF |
| Labeled samples | 3,493,603 (10,927 positive / 3,482,676 negative) |
| Channels used | 18 bipolar |
| Sampling rate | 256 Hz |
| Format | European Data Format (EDF) |

### Channel subset used in benchmarks

Most baselines use the following **18 bipolar channels** for consistency across subjects:

```
FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ,  CZ-PZ
```

### Download

**PhysioNet (free, registration required):**
> https://physionet.org/content/chbmit/1.0.0/

```bash
# Using PhysioNet wget script
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Or using the PhysioNet client
pip install wfdb
python -c "import wfdb; wfdb.dl_database('chbmit', './data/CHB-MIT-scalp-eeg-database-1.0.0')"
```

### Citation

```
Shoeb AH (2009). Application of Machine Learning to Epileptic Seizure Onset Detection
and Treatment. PhD Thesis, MIT.
Goldberger AL et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation 101(23):e215-e220.
```

---

## 2. Siena Scalp EEG Database

**Used by:** Li 2025 (CMFViT cross-subject evaluation) + MV-AFA

### Description

The Siena Scalp EEG Database was collected at the Unit of Neurology and Neurophysiology,
University of Siena, Italy. It contains EEG recordings from **14 adult patients** with
epilepsy, covering a total of 47 seizures. All recordings were performed during standard
video-EEG monitoring sessions; ictal and interictal segments are annotated by clinical
neurophysiologists.

| Property | Value |
|----------|-------|
| Subjects | 14 (adults) |
| Files used | 41 EDF |
| Labeled samples | 9,344 (1,153 positive / 8,191 negative) |
| Channels used | 16 bipolar |
| Sampling rate | 512 Hz |
| Seizure types | Focal, generalised |
| Format | European Data Format (EDF) |

### Download

**PhysioNet (free, registration required):**
> https://physionet.org/content/siena-scalp-eeg/1.0.0/

```bash
# Using PhysioNet wget script
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/

# Or using the PhysioNet client
pip install wfdb
python -c "import wfdb; wfdb.dl_database('siena-scalp-eeg', './data/siena-scalp-eeg-1.0.0')"
```

### Citation

```
Detti P et al. (2020). Siena Scalp EEG Database (version 1.0.0).
PhysioNet. https://doi.org/10.13026/5d4a-j060

Detti P et al. (2020). EEG synchronization analysis for seizure prediction:
A study on data of noninvasive recordings.
Processes 8(7):846. doi: 10.3390/pr8070846
```

---

## 3. SeizeIT2

**Used by:** MV-AFA (multi-center wearable EEG evaluation)

### Description

SeizeIT2 is the first large open dataset of **wearable** data recorded in patients with
focal epilepsy. It comprises more than **11,000 hours of multimodal data** — behind-the-ear
electroencephalography (EEG), electrocardiography (ECG), electromyography (EMG), and movement
signals — collected from **125 patients** across five European Epilepsy Monitoring Centers,
with **883 annotated focal seizures**. Data are stored in Brain Imaging Data Structure (BIDS)
format. In our benchmarks we use the **2-channel behind-the-ear EEG** modality only.

| Property | Value |
|----------|-------|
| Subjects | 125 |
| Files used | 2,846 EDF |
| Labeled samples | 578,390 (13,074 positive / 565,316 negative) |
| Recording centers | 5 (European EMCs) |
| Focal seizures | 883 annotated |
| Channels used | 2 (behind-the-ear EEG) |
| Sampling rate | 256 Hz |
| Modalities | EEG, ECG, EMG, movement |
| Format | BIDS (EDF) |

### Download

**OpenNeuro (free, open access) — accession `ds005873`:**
> https://openneuro.org/datasets/ds005873

```bash
# Via the OpenNeuro CLI
npx @openneuro/cli download --snapshot 1.0.1 ds005873 ./data/seizeit2

# Or via DataLad
datalad install https://github.com/OpenNeuroDatasets/ds005873.git ./data/seizeit2
datalad get ./data/seizeit2
```

### Citation

```
Bhagubai M, Chatzichristos C, Swinnen L, Macea J, Zhang J, Lagae L, Jansen K,
Schulze-Bonhage A, Sales F, Mahler B, Weber Y, Van Paesschen W, De Vos M (2025).
SeizeIT2: Wearable Dataset Of Patients With Focal Epilepsy.
Scientific Data. doi: 10.1038/s41597-025-05580-x
OpenNeuro DOI: 10.18112/openneuro.ds005873.v1.0.1
```

---

## 4. Temple University Hospital EEG Seizure Corpus (TUSZ)

**Used by:** Xu 2026 (TUH), PSD-LW-DCN 2026 (TUSZ) + MV-AFA

### Description

The Temple University Hospital EEG (TUH EEG) corpus is the **largest publicly available
clinical EEG dataset**, collected from routine EEG recordings at Temple University Hospital.
The TUSZ (TUH Seizure) subset provides expert-annotated seizure events with detailed
seizure type labels. Our experiments use a **130-subject subset** under the unified protocol.

| Property | Value |
|----------|-------|
| Subjects used | 130 |
| Files used | 1,237 EDF |
| Labeled samples | 149,133 (19,080 positive / 130,053 negative) |
| Channels used | 20 TCP bipolar |
| Sampling rate | 250–1000 Hz (mixed: 250, 256, 400, 512, 1000 Hz) |
| Seizure types | Multiple (FNSZ, GNSZ, ABSZ, etc.) |
| Format | EDF + TSV annotations |

### Download

**Requires free registration with Temple University:**
> https://isip.piconepress.com/projects/tuh_eeg/

```bash
# After registration, use rsync (credentials provided by TUH)
rsync -auxvL nedc_tuh_eeg@www.isip.piconepress.com:data/tuh_eeg_seizure/ ./data/tusz/
```

> **Note:** The TUSZ dataset requires a data use agreement. Registration is free for
> academic research. Approval typically takes 1–3 business days.

### Version used in benchmarks

Baselines in this repo reference **TUSZ v1.5.1**, restricted to a controlled **130-subject
subset** for reproducible experiments under the unified protocol.

### Citation

```
Obeid I and Picone J (2016). The Temple University Hospital EEG Data Corpus.
Frontiers in Neuroscience 10:196. doi: 10.3389/fnins.2016.00196

Shah V et al. (2018). The Temple University Hospital Seizure Detection Corpus.
Frontiers in Neuroinformatics 12:83. doi: 10.3389/fninf.2018.00083
```

---

## Data Directory Layout (after download)

```
data/
├── README.md                              ← This file
├── CHB-MIT-scalp-eeg-database-1.0.0/     ← CHB-MIT raw EDF files
│   ├── chb01/
│   │   ├── chb01_01.edf
│   │   ├── ...
│   │   └── chb01-summary.txt
│   ├── chb02/
│   └── ...
├── siena-scalp-eeg-1.0.0/                ← Siena EDF files
│   ├── PN00/
│   ├── PN01/
│   └── ...
├── seizeit2/                              ← SeizeIT2 (OpenNeuro ds005873, BIDS)
│   ├── sub-001/
│   ├── sub-002/
│   └── ...
└── tusz/                                  ← TUSZ EDF + annotation files
    ├── train/
    ├── dev/
    └── ...
```

---

## Notes on Data Preprocessing

All preprocessing (bandpass filtering, segmentation, normalisation) is performed **on-the-fly**
inside each baseline script to ensure reproducibility and avoid storing processed derivatives.
See the `--l_freq`, `--h_freq`, `--window_sec`, and `--step_sec` arguments in each script.
