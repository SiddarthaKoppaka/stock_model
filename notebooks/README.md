# DiffSTOCK Training Notebooks

## Google Colab Notebook

### [`DiffSTOCK_Training_Colab.ipynb`](./DiffSTOCK_Training_Colab.ipynb)

Complete end-to-end training notebook for Google Colab with GPU support.

#### Features

- ✅ **One-click setup**: Clones repo, installs dependencies
- ✅ **Data upload**: Upload pre-downloaded dataset or scrape on Colab
- ✅ **Full training pipeline**: Train, validate, test
- ✅ **Google Drive integration**: Saves all outputs to Drive
- ✅ **Comprehensive visualizations**: Training curves, backtest performance
- ✅ **Detailed metrics**: IC, ICIR, Sharpe, Max Drawdown, etc.
- ✅ **Backtesting**: Top-K strategy with realistic Indian market costs
- ✅ **Result export**: Download zip file of all outputs

#### Usage

1. **Open in Colab**:
   - Upload `DiffSTOCK_Training_Colab.ipynb` to Google Colab
   - Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SiddarthaKoppaka/stock_model/blob/main/notebooks/DiffSTOCK_Training_Colab.ipynb)

2. **Select Runtime**:
   - Go to `Runtime` → `Change runtime type`
   - Select `GPU` (T4 recommended)
   - Click `Save`

3. **Run All Cells**:
   - `Runtime` → `Run all`
   - Or use `Ctrl+F9` (Windows/Linux) or `Cmd+F9` (Mac)

4. **Upload Dataset**:
   - When prompted, upload these files from your local `data/dataset/`:
     - `nifty500_10yr.npz`
     - `relation_matrices.npz`
   - **OR** uncomment the scraping cells (takes 30-60 min)

5. **Wait for Training**:
   - Training takes 2-4 hours on GPU
   - All progress is logged in real-time
   - Checkpoints saved every 10 epochs

6. **Review Results**:
   - Training curves plotted automatically
   - Validation and test metrics computed
   - Backtest results with portfolio visualization
   - All outputs saved to Google Drive

#### Outputs Saved

All outputs are saved to: `Google Drive/DiffSTOCK_Outputs/run_YYYYMMDD_HHMMSS/`

```
run_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_model.pt
│   └── final_model.pt
├── logs/
│   └── training_history.json
├── training_summary.json
├── training_curves.png
├── validation_metrics.json
├── validation_results.npz
├── test_metrics.json
├── test_results.npz
├── backtest_summary.json
├── backtest_results.npz
├── backtest_performance.png
└── REPORT.txt
```

#### Training Metrics Tracked

- **Training Loss**: Diffusion loss per epoch
- **Validation IC**: Information Coefficient (rank correlation)
- **Validation ICIR**: IC Information Ratio (consistency)
- **Validation Accuracy**: Direction prediction accuracy
- **Learning Rate**: Cosine annealing schedule

#### Evaluation Metrics

- **IC** (Information Coefficient): Target > 0.04
- **ICIR** (IC Information Ratio): Target > 0.3
- **Rank IC**: Rank-based correlation
- **Accuracy**: Direction prediction (up/down)
- **MCC** (Matthews Correlation Coefficient)

#### Backtest Metrics

- **Total Return**: Overall portfolio return
- **Annualized Return**: Yearly return
- **Sharpe Ratio**: Risk-adjusted returns (target > 1.0)
- **Max Drawdown**: Peak-to-trough decline
- **Win Rate**: % of positive days
- **Avg Turnover**: Portfolio rebalancing rate

#### Hyperparameter Tuning

The notebook includes a section with tuning suggestions based on results:

- **Low IC** → Increase model capacity
- **Overfitting** → Add regularization
- **Instability** → Reduce learning rate
- **Underfitting** → Increase model complexity

Edit `config/config.yaml` and retrain!

#### Expected Results

| Metric | Expected Range |
|--------|----------------|
| Train Loss (final) | 0.004-0.008 |
| Val IC | 0.04-0.07 |
| Test IC | 0.02-0.05 |
| Sharpe Ratio | 1.0-1.5 |
| Max Drawdown | -15% to -25% |

#### Requirements

- **Google Account**: For Colab and Drive
- **GPU Runtime**: T4 (free) or better
- **Dataset**: 2 files (~500MB total)
- **Time**: 2-4 hours for training

#### Troubleshooting

**"CUDA out of memory"**:
- Reduce batch size in config (32 → 16)
- Reduce model size (d_model: 128 → 64)
- Use CPU runtime (slower)

**"Dataset not found"**:
- Ensure you uploaded both `.npz` files
- Check they're in `data/dataset/` directory

**"Training diverged"**:
- Reduce learning rate (3e-4 → 1e-4)
- Increase warmup steps (1000 → 2000)
- Check data quality

#### Tips

1. **Save checkpoints**: Models are saved to Google Drive automatically
2. **Monitor progress**: Watch training curves in real-time
3. **Early stopping**: Training stops if no improvement for 20 epochs
4. **Download results**: Use the download cell to get a zip file
5. **Multiple runs**: Each run gets a unique timestamp folder

#### Next Steps After Training

1. **Analyze metrics**: Review test IC and Sharpe ratio
2. **Tune hyperparameters**: Adjust config based on results
3. **Ensemble models**: Train multiple models with different seeds
4. **Deploy**: Integrate best model with broker API

## Local Jupyter Notebooks (Optional)

You can also create local notebooks for:
- `01_data_exploration.ipynb`: EDA on downloaded data
- `02_training_monitor.ipynb`: Live training monitoring
- `03_model_analysis.ipynb`: Prediction analysis

Use the Colab notebook as a template!

---

**Note**: The Colab notebook is self-contained and can run independently. All you need is the dataset files and a Google account.

**Happy training! 📈🚀**
