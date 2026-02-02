# cal_prop
This repository provides a **TensorFlow/Keras ensemble inference tool** for predicting
vapor pressure and key thermodynamic properties of pure substances from molecular
representations.

The tool is designed for **research and reproducible inference**, supporting:
- Multiple SMILES
- Multiple temperatures or reduced temperatures
- Ensemble uncertainty (mean ± std)
- Legacy TensorFlow 2.3.0 compatibility

---

## Key Features

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

## Supported Tasks

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

## Usage

### Single SMILES, multiple temperatures (no CSV)

#### Real vapor pressure
```bash
python predict_realt_multi.py \
  --task both_real --rep FP \
  --smiles "CCCCCC(C)Br" \
  --temps 298.15,320,350 \
  --i-list 1-10 --j-list 1-3 \
  --out-csv out_single_real.csv
