"""Single source of truth for the asset universe and train/test split."""

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "TLT", "GLD", "VNQ"]
PROFILES = ["conservative", "balanced", "aggressive"]

DATA_START = "2013-01-01"
TRAIN_END = "2020-12-31"
OOS_START = "2021-01-01"

LOOKBACK = 30
N_ASSETS = len(TICKERS)
