# 🔬 Vapor Pressure & Thermodynamic Property Ensemble Predictor

This repository provides a **TensorFlow/Keras ensemble inference tool** for predicting
vapor pressure and key thermodynamic properties of pure substances from molecular
representations.

The tool is designed for **research and reproducible inference**, supporting:
- Multiple SMILES
- Multiple temperatures or reduced temperatures
- Ensemble predictions with mean and standard deviation
- HGB-based expected error estimation
- Legacy TensorFlow 2.3.0 compatibility

---

## ✨ Key Features

- Ensemble inference over multiple trained models
- Supports **real** and **reduced** vapor pressure formulations
- Predicts:
  - Vapor pressure
  - Tb, Tc, lnPc, ω
- Reports:
  - Ensemble mean prediction
  - Ensemble standard deviation
  - HGB predicted expected absolute error
- Flexible input:
  - Command-line SMILES
  - CSV files (`smiles + T(K)` or `smiles + Tr`)
- Deterministic output ordering  
  (input SMILES order, temperatures sorted ascending)
- Automatic terminal anchor at `Tr = 1, lnPr = 0` for pvap shift correction

---

## 📦 Supported Tasks

| Task name | Description |
|----------|------------|
| `pvap_reduced` | Predict **reduced vapor pressure** `lnPr = ln(P / Pc)` |
| `pvap_real` | Predict **real vapor pressure** `lnP (Pa)` |
| `TbTcPcw` | Predict **thermodynamic properties only**: Tb, Tc, lnPc, ω |
| `both_reduced` | Predict **lnPr + Tb + Tc + lnPc + ω** |
| `both_real` | Predict **lnP + Tb + Tc + lnPc + ω** |

**Note:**  
`TbTcPcw` can be used independently or as the base step for `*_real` vapor pressure tasks,
where predicted Tc is used to reduce real temperature.

---

## 🛠 Installation

### Option 1: Installation via PyPI (recommended for inference)

```bash
pip install numpy==1.18.5 pandas==1.3.5 openpyxl==3.1.3 tensorflow==2.3.0
```

Optional (legacy GPU setup):
```bash
pip install tensorflow-gpu==2.3.0
```

> Notes:
> - TensorFlow 2.3.0 requires `numpy < 1.19.0`
> - GPU inference requires compatible CUDA/cuDNN (not included)

---

### Option 2: Installation with Conda (recommended for reproducibility)

```bash
conda env create -f environment.yml
conda activate cal_prop
```

Verify installation:
```bash
python - << EOF
import sys, numpy, pandas, openpyxl, tensorflow as tf
print("Python:", sys.version)
print("TF:", tf.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("openpyxl:", openpyxl.__version__)
EOF
```

Expected versions:
- Python 3.7.16
- TensorFlow 2.3.0
- NumPy 1.18.5
- Pandas 1.3.5
- openpyxl 3.1.3

---

## 🚀 Usage

### Option A: Run with a Python command

Example (real vapor pressure + properties, CSV input):

```bash
python predict.py \
  --xlsx-path input_features.xlsx \
  --input-csv example.csv --smiles-col smiles --temp-col "T(K)" \
  --model-dir ./model_save \
  --error-model-dir ./model_save \
  --stats-json normalization_stats.json \
  --rep FP \
  --task both_real \
  --i-list 1-10 \
  --j-list 1-10 \
  --out-csv output.csv \
  --progress
```

>
> Note: for `both_real` / `pvap_real`, the CSV should contain a **temperature column in Kelvin** (e.g. `T(K)`).

### Option B: Run using the provided shell script (`predict.sh`)

For convenience, this repository provides a wrapper script `predict.sh` which runs an
equivalent command.

#### Step 1: Make the script executable
```bash
chmod +x predict.sh
```

#### Step 2: Run
```bash
./predict.sh
```

The current `predict.sh` content is:

```bash
python predict.py \
  --xlsx-path input_features.xlsx \
  --input-csv example.csv --smiles-col smiles --temp-col "T(K)" \
  --model-dir ./model_save \
  --error-model-dir ./model_save \
  --stats-json normalization_stats.json \
  --rep FP \
  --task both_real \
  --i-list 1-10 \
  --j-list 1-10 \
  --out-csv output.csv \
  --progress
```

Edit `predict.sh` to customize `--task`, `--rep`, input CSV path/column names, ensemble ranges, and output path.

## 📊 Output and Expected Error

The ensemble models provide the mean prediction and standard deviation across
individual models. In addition, pretrained HistGradientBoosting (HGB) models
estimate the expected absolute prediction error.

For thermodynamic properties, the output includes:

- `Tb_mean`, `Tb_std`, `Tb_expected_error`
- `Tc_mean`, `Tc_std`, `Tc_expected_error`
- `lnPc_mean`, `lnPc_std`, `lnPc_expected_error`
- `w_mean`, `w_std`, `w_expected_error`

For vapor pressure predictions, the corresponding prediction, ensemble standard
deviation, and expected error are also reported.

The expected error estimates the possible error of each prediction. A smaller value generally indicates a more reliable prediction. It is provided as a reliability reference, not as a confidence interval.

## 📁 Repository Structure

| File / Folder | Description |
|--------------|-------------|
| `predict.py` | Main inference script for ensemble prediction and expected error estimation |
| `predict.sh` | Convenience shell script wrapping common prediction commands |
| `environment.yml` | Reproducible Conda environment definition |
| `requirements.txt` | Python package requirements |
| `input_features.xlsx` | Input feature data |
| `normalization_stats.json` | Normalization statistics for model inputs/outputs |
| `model_save/` | Trained Keras ensemble models and portable HGB expected error models |
| `example.csv` | Example input for prediction |
| `README.md` | Project documentation |

## 📜 Citation

This work is currently under preparation for journal publication.

If you use this code before the paper is published, please cite as:

```
Author(s), "Title (in preparation)", Journal, Year.
```

The BibTeX entry will be provided once the paper is accepted.

---

## 📚 References

The molecular fingerprints (FP) used in this repository are derived from the
directed message passing neural network (D-MPNN) framework described in the
following work:

1. Yen-Hsiang Lin, Hsin-Hao Liang, Shiang-Tai Lin, Yi-Pei Li,  
   *Advancing vapor pressure prediction: A machine learning approach with directed message passing neural networks*,  
   **Journal of the Taiwan Institute of Chemical Engineers**, 2024.  
   https://doi.org/10.1016/j.jtice.2024.105926

This repository does **not** redistribute the original publication or its
supplementary materials. Users are encouraged to consult the original article
for detailed descriptions of the D-MPNN architecture, fingerprint construction,
and training methodology.

---

## 🧪 Tested Environment

- Python 3.7.16
- TensorFlow 2.3.0
- NumPy 1.18.5
- Pandas 1.3.5
- openpyxl 3.1.3

A fully reproducible Conda environment is provided.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Maintained By

Maintained by HsingHao Liang ([@HsingHaoLiang](https://github.com/HsingHaoLiang)).  
COMET, Department of Chemical Engineering, National Taiwan University
