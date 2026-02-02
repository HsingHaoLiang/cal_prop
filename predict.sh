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
