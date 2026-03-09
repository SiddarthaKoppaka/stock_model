"""
HMM Regime Detector

Fits a 4-state Gaussian HMM on NIFTY 50 returns to identify latent market
regimes. Regimes are labelled for every trading day and stored as soft
probability vectors that are fed to the model as conditioning signals.

CRITICAL: HMM is FIT only on training period (≤ train_end_date) to prevent
lookahead leakage. The fitted model then labels all dates including val/test.

Outputs:
    data/regime/hmm_model.pkl              — fitted hmmlearn GaussianHMM
    data/regime/daily_regime_probs.parquet — (date, prob_state_0..3)
    data/regime/regime_labels.png          — NIFTY 50 price colored by regime

Usage:
    cd diffstock_india
    .venv/bin/python -m src.data.regime_detector
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.error("hmmlearn not installed. Run: pip install hmmlearn")


# ─── constants ────────────────────────────────────────────────────────────────
NIFTY50_TICKER  = "^NSEI"
FETCH_START     = "2004-01-01"   # 1 year before 2005 for warm-up indicators
N_STATES        = 4
N_ITER          = 500
RANDOM_STATE    = 42


def _build_hmm_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 5-feature matrix for HMM from NIFTY 50 OHLCV data.

    Features (all computed causally):
      - ret_5d    : 5-day rolling return
      - ret_20d   : 20-day rolling return
      - vol_20d   : 20-day annualised volatility
      - vol_60d   : 60-day annualised volatility
      - vol_ratio : vol_20d / vol_60d  (regime-transition indicator)
    """
    close = nifty_df['Close']
    log_ret = np.log(close / close.shift(1))

    df = pd.DataFrame(index=nifty_df.index)
    df['ret_5d']    = close.pct_change(5)
    df['ret_20d']   = close.pct_change(20)
    df['vol_20d']   = log_ret.rolling(20).std() * np.sqrt(252)
    df['vol_60d']   = log_ret.rolling(60).std() * np.sqrt(252)
    df['vol_ratio'] = df['vol_20d'] / (df['vol_60d'] + 1e-8)

    return df.dropna()


class RegimeDetector:
    """
    Fits a Gaussian HMM on NIFTY 50 returns to detect latent market regimes.

    4 recommended states for Indian market:
      State 0 — Low-vol bull (typical 2010–2019 conditions)
      State 1 — High-vol bull (post-dip recovery, momentum)
      State 2 — High-vol bear (GFC 2008, COVID 2020)
      State 3 — Low-vol consolidation (range-bound, uncertain)

    The mapping of state indices to regimes is determined by the fitted HMM's
    emission means (state with highest vol_20d mean = bear regime).
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        n_iter: int = N_ITER,
        output_dir: Path = Path("data/regime"),
        train_end_date: str = "2022-12-31",
        random_state: int = RANDOM_STATE,
    ):
        self.n_states       = n_states
        self.n_iter         = n_iter
        self.output_dir     = Path(output_dir)
        self.train_end_date = pd.Timestamp(train_end_date)
        self.random_state   = random_state
        self.model: Optional[GaussianHMM] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── data fetching ──────────────────────────────────────────────────────────

    def fetch_nifty50(self) -> pd.DataFrame:
        """Download NIFTY 50 index data from yfinance."""
        logger.info(f"Downloading {NIFTY50_TICKER} from {FETCH_START}...")
        df = yf.download(NIFTY50_TICKER, start=FETCH_START, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"Failed to download {NIFTY50_TICKER}")
        df.index = pd.to_datetime(df.index)
        logger.info(f"Downloaded {len(df)} days of NIFTY 50 data")
        return df

    # ── fitting ────────────────────────────────────────────────────────────────

    def fit(self, nifty_df: pd.DataFrame) -> None:
        """
        Fit GaussianHMM on training period features.

        Args:
            nifty_df: NIFTY 50 OHLCV DataFrame (yfinance format)
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn required. Run: pip install hmmlearn")

        features = _build_hmm_features(nifty_df)

        # Restrict to training period only — no future leakage
        train_features = features[features.index <= self.train_end_date]
        logger.info(
            f"Fitting HMM on {len(train_features)} days "
            f"({train_features.index[0].date()} → {train_features.index[-1].date()})"
        )

        X = train_features.values  # (T, 5)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X)
        logger.info(f"HMM fitted | converged={self.model.monitor_.converged}")
        self._log_regime_stats(train_features)

    def _log_regime_stats(self, features: pd.DataFrame) -> None:
        """Log per-state emission means for interpretability."""
        X = features.values
        states = self.model.predict(X)
        df_s = features.copy()
        df_s['state'] = states

        logger.info("─── HMM Regime Statistics (training period) ───")
        for s in range(self.n_states):
            mask = df_s['state'] == s
            if mask.sum() == 0:
                continue
            pct = 100 * mask.mean()
            vol = df_s.loc[mask, 'vol_20d'].mean()
            ret = df_s.loc[mask, 'ret_20d'].mean()
            logger.info(
                f"  State {s}: {mask.sum():4d} days ({pct:.1f}%)  "
                f"vol_20d={vol:.3f}  ret_20d={ret:+.3f}"
            )

    # ── labelling ──────────────────────────────────────────────────────────────

    def get_regime_labels(self, nifty_df: pd.DataFrame) -> pd.DataFrame:
        """
        Label ALL dates (including val/test) using the training-fitted model.

        Returns:
            DataFrame with columns [date, prob_state_0, ..., prob_state_{N-1}]
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = _build_hmm_features(nifty_df)
        X = features.values

        # Soft (posterior) probabilities
        log_probs = self.model.predict_proba(X)  # (T, n_states)

        df_probs = pd.DataFrame(
            log_probs,
            index=features.index,
            columns=[f"prob_state_{i}" for i in range(self.n_states)],
        )
        df_probs.index.name = "date"
        df_probs = df_probs.reset_index()

        logger.info(
            f"Regime labels computed for {len(df_probs)} days "
            f"({df_probs['date'].min().date()} → {df_probs['date'].max().date()})"
        )
        return df_probs

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save fitted HMM model to disk."""
        model_path = self.output_dir / "hmm_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"HMM model saved → {model_path}")

    def load(self) -> None:
        """Load a previously saved HMM model."""
        model_path = self.output_dir / "hmm_model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"HMM model loaded from {model_path}")

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot_regimes(self, nifty_df: pd.DataFrame, regime_df: pd.DataFrame) -> None:
        """
        Save regime_labels.png: NIFTY 50 price coloured by dominant regime.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            colors = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db"]  # green, orange, red, blue
            labels = ["Low-vol bull", "High-vol bull", "High-vol bear", "Consolidation"]

            close = nifty_df["Close"].reindex(pd.to_datetime(regime_df["date"])).dropna()
            dates = pd.to_datetime(regime_df.set_index("date").index)
            dominant = regime_df.set_index("date")[
                [f"prob_state_{i}" for i in range(self.n_states)]
            ].idxmax(axis=1).str.extract(r"(\d+)")[0].astype(int)
            dominant = dominant.reindex(close.index)

            fig, ax = plt.subplots(figsize=(16, 5))
            for i in range(len(close) - 1):
                if dominant.iloc[i] is not None:
                    c = colors[int(dominant.iloc[i])]
                    ax.fill_between(
                        [close.index[i], close.index[i + 1]],
                        0, close.iloc[i],
                        color=c, alpha=0.4
                    )
            ax.plot(close.index, close.values, color="black", linewidth=0.8)
            ax.set_title("NIFTY 50 — HMM Regime Labels")
            ax.set_ylabel("NIFTY 50 Index")
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(self.n_states)]
            ax.legend(handles=patches, loc="upper left")
            plt.tight_layout()
            out = self.output_dir / "regime_labels.png"
            plt.savefig(out, dpi=120)
            plt.close()
            logger.info(f"Regime plot saved → {out}")
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")

    # ── full pipeline ─────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Full pipeline: fetch → fit → label → save.

        Returns:
            DataFrame with daily regime probabilities
        """
        nifty_df = self.fetch_nifty50()
        self.fit(nifty_df)
        regime_df = self.get_regime_labels(nifty_df)

        # Save parquet
        parquet_path = self.output_dir / "daily_regime_probs.parquet"
        regime_df.to_parquet(parquet_path, index=False)
        logger.info(f"Regime probs saved → {parquet_path}")

        self.save()
        self.plot_regimes(nifty_df, regime_df)

        return regime_df


# ─── entry point ──────────────────────────────────────────────────────────────

def fit_regime_model(config: dict) -> pd.DataFrame:
    """Entry point called from dataset build pipeline."""
    root = Path(config["paths"]["root"])
    output_dir = root / config["data"].get("regime_probs_path", "data/regime/daily_regime_probs.parquet")
    output_dir = output_dir.parent  # get the directory

    detector = RegimeDetector(
        n_states=config.get("model", {}).get("n_regime_states", 4),
        output_dir=output_dir,
        train_end_date=config["training"]["train_end"],
    )
    return detector.run()


if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.utils.logger import setup_logger
    setup_logger(log_dir=Path(config["paths"]["root"]) / "logs", log_level="INFO")

    regime_df = fit_regime_model(config)
    print(regime_df.head())
    print(f"\nTotal days labelled: {len(regime_df)}")
