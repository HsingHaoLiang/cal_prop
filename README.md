# 🔬 Vapor Pressure & Thermodynamic Property Ensemble Predictor

This repository provides a **TensorFlow/Keras ensemble inference tool** for predicting
vapor pressure and key thermodynamic properties of pure substances from molecular
representations.

The tool is designed for **research and reproducible inference**, supporting:
- Multiple SMILES
- Multiple temperatures or reduced temperatures
- Ensemble uncertainty (mean ± std)
- Legacy TensorFlow 2.3.0 compatibility

---

## ✨ Key Features

- Ensemble inference over multiple trained models
- Supports **real** and **reduced** vapor pressure formulations
- Predicts:
  - Vapor pressure
  - Tb, Tc, lnPc, ω
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
pip install numpy==1.18.5 pandas==1.1.5 openpyxl==3.0.10 tensorflow==2.3.0
```

Optional (legacy GPU setup):
```bash
pip install tensorflow-gpu==2.3.0
```

> Notes:
> - TensorFlow 2.3.0 requires `numpy < 1.19.0`
> - Do **not** install `tensorflow` and `tensorflow-gpu` at the same time
> - GPU inference requires compatible CUDA/cuDNN (not included)

---

### Option 2: Installation from Source (Conda, fully reproducible)

```bash
conda env create -f environment.yml
conda activate pvap_infer_tf23_cpu
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
- Pandas 1.1.5
- openpyxl 3.0.10

---

## 🚀 Usage

### Option A: Run with a Python command

Example (real vapor pressure + properties, CSV input):

```bash
python predict.py \
  --xlsx-path input_features.xlsx \
  --smiles-csv smiles_Tr.csv --smiles-col smiles --temp-col "T(K)" \
  --model-dir ./model_save \
  --stats-json normalization_stats.json \
  --rep FP \
  --task both_real \
  --i-list 1-10 \
  --j-list 1-10 \
  --out-csv output.csv
```

> If your script uses `--tr-col "T(K)"` for a real-temperature task, rename it to `--temp-col "T(K)"` (or adjust according to the argument name in your `predict.py`).
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
  --smiles-csv smiles_Tr.csv --smiles-col smiles --tr-col "T(K)" \
  --model-dir ./model_save \
  --stats-json normalization_stats.json \
  --rep FP \
  --task both_real \
  --i-list 1-10 \
  --j-list 1-10 \
  --out-csv output.csv
```

Edit `predict.sh` to customize `--task`, `--rep`, input CSV path/column names, ensemble ranges, and output path.

## 📁 Repository Structure

```text
.
├── predict.py                 # main inference script
├── predict.sh                 # convenience shell wrapper
├── environment.yml             # reproducible environment
├── normalization_stats.json
├── model_save/                 # trained Keras models
├── smiles_Tr.csv               # example CSV input
└── README.md
```

---

## 📜 Citation

This work is currently under preparation for journal publication.

If you use this code before the paper is published, please cite as:

```
Author(s), "Title (in preparation)", Journal, Year.
```

The BibTeX entry will be provided once the paper is accepted.

---

## 🧪 Tested Environment

- Python 3.7.16
- TensorFlow 2.3.0
- NumPy 1.18.5
- Pandas 1.1.5
- openpyxl 3.0.10

A fully reproducible Conda environment is provided.
