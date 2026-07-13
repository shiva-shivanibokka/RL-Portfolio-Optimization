"""One-time fetch of REAL adjusted-close prices -> data/prices.parquet.
Run manually: python -m rl.data_pull
Training deps only (yfinance). Not imported at serve time."""
import sys
import pandas as pd
from data.UNIVERSE import TICKERS, DATA_START


def fetch_prices(tickers, start, end=None) -> pd.DataFrame:
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close[tickers]  # enforce column order
        if close.dropna(how="all").empty:
            raise ValueError("yfinance returned no data")
    except Exception as e:
        print(f"yfinance failed ({e}); falling back to Stooq", file=sys.stderr)
        from pandas_datareader import data as pdr
        cols = {}
        for t in tickers:
            s = pdr.DataReader(t, "stooq", start=start, end=end)["Close"].sort_index()
            cols[t] = s
        close = pd.DataFrame(cols)
        close = close[tickers]

    close = close.asfreq("B").ffill().dropna()
    if close.isna().any().any():
        raise ValueError("NaNs remain after ffill/dropna")
    return close


def main():
    prices = fetch_prices(TICKERS, DATA_START)
    if len(prices) < 1500:
        print(f"WARNING: only {len(prices)} rows fetched", file=sys.stderr)
    prices.to_parquet("data/prices.parquet")
    print(f"Wrote data/prices.parquet: {prices.shape}, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}")


if __name__ == "__main__":
    main()
