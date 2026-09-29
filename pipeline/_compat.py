"""
Parquet compatibility shim.

If pyarrow/fastparquet are not installed, patches pd.read_parquet and
DataFrame.to_parquet to use pickle so the pipeline can run without them.
Pickle files use the same .parquet extension; once pyarrow is installed
run process_all.py --rebuild to regenerate proper parquet files.
"""
import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        HAS_PARQUET = True
    except ImportError:
        HAS_PARQUET = False


if not HAS_PARQUET:
    _orig_read_parquet = pd.read_parquet

    def _read_parquet_compat(path, **kwargs):
        try:
            return pd.read_pickle(str(path))
        except Exception:
            return _orig_read_parquet(path, **kwargs)

    def _to_parquet_compat(self, path, index=True, **kwargs):
        df = self if index else self.reset_index(drop=True)
        df.to_pickle(str(path))

    pd.read_parquet = _read_parquet_compat
    pd.DataFrame.to_parquet = _to_parquet_compat
