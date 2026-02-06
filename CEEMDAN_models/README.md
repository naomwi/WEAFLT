# CEEMDAN-based Water Quality Forecasting

Simple, clean implementation for predicting water quality (EC - Electrical Conductivity) using CEEMDAN decomposition and deep learning models.

## Features

- **CEEMDAN Decomposition**: Decomposes EC time series into 12 IMFs + 1 residue
- **3 Deep Learning Models**: LSTM, PatchTST, Transformer
- **6 Prediction Horizons**: 6, 12, 24, 48, 96, 168 hours
- **Simple Implementation**: No advanced features - standard MSE loss, basic early stopping
- **Saves All IMFs**: IMF components saved to files for inspection

## Project Structure

```
CEEMDAN_models/
├── config.py              # Configuration settings
├── main.py                # Main entry point
├── train.py               # Training module
├── evaluate.py            # Evaluation module
├── requirements.txt       # Dependencies
├── models/
│   ├── lstm.py            # LSTM model
│   ├── patchtst.py        # PatchTST model
│   └── transformer.py     # Transformer model
├── utils/
│   ├── ceemdan_decomposition.py   # CEEMDAN implementation
│   ├── data_loader.py             # Data loading utilities
│   └── metrics.py                 # Evaluation metrics
├── data/                  # Raw data directory
├── decomposed_imfs/       # Saved IMF files
├── saved_models/          # Trained model weights
└── results/
    ├── plots/             # Visualization outputs
    └── metrics/           # Evaluation results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your water quality data CSV in the `data/` directory or ensure it's accessible from one of the configured paths.

### 3. Run Full Pipeline

```bash
python main.py
```

This will:
1. Load and decompose the EC data using CEEMDAN
2. Train LSTM, PatchTST, and Transformer models for each IMF component
3. Evaluate all models across all horizons
4. Generate comparison reports and plots

### 4. Run Specific Steps

```bash
# Only run CEEMDAN decomposition
python main.py --decompose-only

# Only train models (requires decomposed data)
python main.py --train-only

# Only evaluate (requires trained models)
python main.py --evaluate-only
```

## Configuration

Edit `config.py` to modify:

- **CEEMDAN settings**: Number of IMFs, noise width, trials
- **Data settings**: Target column, train/val/test ratios, sequence length
- **Model settings**: Hidden sizes, number of layers, dropout
- **Training settings**: Epochs, learning rate, early stopping patience

## Metrics

The following metrics are computed:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

## Output Files

After running the full pipeline:

- `decomposed_imfs/ec_imf_*.npy`: Individual IMF components
- `decomposed_imfs/ec_residue.npy`: Residue component
- `saved_models/*.pth`: Trained model weights
- `results/metrics/full_evaluation_results.csv`: Complete results
- `results/metrics/summary_results.csv`: Summary table
- `results/plots/*.png`: Visualization plots

## Design Principles

This implementation follows a **simple, clean** design:

- ✅ Standard MSE loss (no weighted loss)
- ✅ Basic early stopping
- ✅ Separate model per IMF component
- ✅ Simple summation for reconstruction
- ❌ No complex learning rate schedulers
- ❌ No attention mechanisms between IMFs
- ❌ No multi-task learning
