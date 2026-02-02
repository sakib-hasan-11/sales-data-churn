from typing import Any, Dict, Optional
import pandas as pd



def raw_preprocess(
    df: pd.DataFrame,
    strategy: str = "auto",
    custom_fill: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Fill missing values in a DataFrame.

    Args:
        df: DataFrame to process (modified copy returned)
        strategy: 'auto' or 'constant' or 'median' or 'mode'. In 'auto' we use heuristics:
           - numeric: median
           - object/category: mode or 'Unknown'
           - datetime: forward fill then backward fill
        custom_fill: optional mapping of column -> fill value which overrides strategy

    Returns:
        DataFrame with missing values filled.
    """
    # name clearning 
    def _clean(col: str) -> str:
        col = col.strip()
        col = col.replace("/", "_")
        col = col.replace(" ", "_")
        # keep alnum and underscore
        import re

        col = re.sub(r"[^0-9a-zA-Z_]+", "", col)
        col = re.sub(r"_+", "_", col)
        col = col.lower()
        return col
    new_cols = {c: _clean(c) for c in df.columns}
    df = df.rename(columns=new_cols)





    # fill nulls 
    out = df.copy()

    if custom_fill is None:
        custom_fill = {}

    # first apply custom fills
    for col, val in custom_fill.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)

    # apply strategy to remaining nulls
    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue

        if col in custom_fill:
            continue

        dtype = out[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            if strategy == "constant":
                fill = 0
            elif strategy == "median":
                fill = out[col].median()
            elif strategy == "mode":
                fill = out[col].mode().iloc[0] if not out[col].mode().empty else 0
            else:  # auto
                fill = out[col].median()
            out[col] = out[col].fillna(fill)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            # try forward then backward fill
            out[col] = out[col].fillna(method="ffill").fillna(method="bfill")
        else:
            # treat as categorical / object
            if strategy == "constant":
                fill = "Unknown"
            elif strategy == "mode":
                fill = (
                    out[col].mode().iloc[0] if not out[col].mode().empty else "Unknown"
                )
            else:  # auto or median
                # prefer mode if available else Unknown
                fill = (
                    out[col].mode().iloc[0] if not out[col].mode().empty else "Unknown"
                )
            out[col] = out[col].fillna(fill)

    return out





