from __future__ import annotations

import warnings

import pandas as pd


# Uber's NYSE IPO date. Any rows before this are data errors.
_UBER_IPO_DATE = pd.Timestamp("2019-05-10")
_CSV_DATE_FORMAT = "%d-%m-%Y"


def load_and_clean_stock_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load, clean, and return UBER stock price data as a DatetimeIndex DataFrame.

    Date format
    -----------
    The CSV uses DD-MM-YYYY (e.g. "10-05-2019" = May 10 2019).
    We parse with the explicit format string "%d-%m-%Y" to prevent pandas
    from misreading ambiguous dates such as "10-05-2019" as October 5.

    Design decisions
    ----------------
    * Only ACTUAL trading days present in the CSV are kept.
      bdate_range / ffill is NOT used — that inserts phantom holiday rows.
    * Rows before the UBER IPO date (2019-05-10) are dropped as a safety net.
    * A calendar-gap warning is raised for gaps > 5 calendar days.
    """
    if csv_path is None:
        from uber_stock.config import DATA_RAW
        csv_path = DATA_RAW / "uber_stock_data.csv"

    df = pd.read_csv(csv_path)

    # ── Standardize column names ───────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── Parse dates with the EXACT format from the CSV ────────────────────
    # The CSV has DD-MM-YYYY (e.g. "10-05-2019").
    # format="mixed" / dayfirst=False misread "10-05-2019" as Oct 5,
    # scrambling price order and corrupting all downstream returns.
    df["Date"] = pd.to_datetime(
        df["Date"],
        format=_CSV_DATE_FORMAT,
        errors="coerce",
    )
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        warnings.warn(
            f"{invalid_dates} rows had dates that could not be parsed with "
            f"format '{_CSV_DATE_FORMAT}' and were dropped. "
            "If your CSV uses a different date format, update _CSV_DATE_FORMAT.",
            UserWarning,
            stacklevel=2,
        )
    df = df.dropna(subset=["Date"]).copy()

    # ── Deduplicate and sort chronologically ──────────────────────────────
    df = (
        df.drop_duplicates(subset=["Date"])
          .sort_values("Date")
          .reset_index(drop=True)
    )

    # ── Drop pre-IPO rows (safety net) ────────────────────────────────────
    pre_ipo_mask = df["Date"] < _UBER_IPO_DATE
    n_pre_ipo = int(pre_ipo_mask.sum())
    if n_pre_ipo > 0:
        warnings.warn(
            f"{n_pre_ipo} row(s) found before UBER IPO date "
            f"({_UBER_IPO_DATE.date()}): "
            f"{df.loc[pre_ipo_mask, 'Date'].dt.date.tolist()}. "
            "Dropping these rows.",
            UserWarning,
            stacklevel=2,
        )
        df = df[~pre_ipo_mask].reset_index(drop=True)

    # ── Ensure numeric columns ─────────────────────────────────────────────
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows where any required column is NaN ─────────────────────────
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        required_cols.append("Adj Close")
    df = df.dropna(subset=required_cols)

    # ── Set price column ───────────────────────────────────────────────────
    if "Adj Close" in df.columns:
        df["price"] = df["Adj Close"]
    else:
        df["price"] = df["Close"]

    # ── Keep only positive prices and positive volume ──────────────────────
    df = df[(df["price"] > 0) & (df["Volume"] > 0)].copy()

    # ── Minimum-row guard ────────────���────────────────────────────────────
    if len(df) < 63:
        raise ValueError(
            f"Only {len(df)} valid trading-day rows remain after cleaning. "
            "Need at least 63 for rolling-volatility features. "
            "Check the CSV date format and data quality."
        )

    # ── Set DatetimeIndex ─────────────────────────────────────────────────
    df = df.set_index("Date")
    df.index.name = "Date"

    # ── Warn about calendar gaps longer than 5 days ────────────────────────
    if len(df) > 1:
        day_gaps   = df.index.to_series().diff().dt.days.dropna()
        large_gaps = day_gaps[day_gaps > 5]
        if not large_gaps.empty:
            worst_gap  = int(large_gaps.max())
            worst_date = large_gaps.idxmax().date()
            warnings.warn(
                f"{len(large_gaps)} gap(s) > 5 calendar days detected. "
                f"Largest: {worst_gap} days ending {worst_date}. "
                "No forward-fill applied — only actual trading days are kept.",
                UserWarning,
                stacklevel=2,
            )

    return df