import pandas as pd
from pathlib import Path
import re
import logging
from functionalities.common import infer_year_from_filename, norm_key

logger = logging.getLogger(__name__)

# ---------- helpers ----------

def normalize_rank(raw) -> str:
    if raw is None or pd.isna(raw):
        return "Unranked"
    text = str(raw).strip()
    return " ".join(word[0].upper() + word[1:] if word else "" for word in text.split())

def collect_for_codes(row, start_col_idx: int) -> str:
    """Collect FOR codes from trailing columns without assuming 4 digits.
    Splits on common delimiters and de-dups while preserving order."""
    NA_TOKENS = {"", "N/A", "NA", "NAN", "NAT", "NONE", "NULL"}
    vals = []

    for v in row.iloc[start_col_idx:]:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.upper() in NA_TOKENS:
            continue

        parts = re.split(r"[;,/|\s]+", s)
        for p in parts:
            t = p.strip()
            if t and t.upper() not in NA_TOKENS:
                vals.append(t)

    seen, out = set(), []
    for c in vals:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return ",".join(out) if out else "N/A"

# ---------- core file parser ----------

CORE_SYNONYMS = {
    "name": {"name", "title", "conference name", "conference title"},
    "acronym": {"acronym", "short name"},
    "source": {"source", "database"},
    "rank": {"rank", "rating"},
}

def _map_core_columns(df_cols):
    mapped = {}
    for col in df_cols:
        coln = str(col).lower().strip()
        for canon, alts in CORE_SYNONYMS.items():
            if coln in alts:
                mapped[col] = canon
                break
    return mapped

def load_core_csv(path: Path) -> pd.DataFrame:
    """
    Load CORE rankings from CSV using synonym-based mapping.
    
    Format Assumptions:
    - Files are exports from the CORE Conference Rankings Portal.
    - Standard CORE exports often lack headers, so fallback to positional indexing 
      is used (Col 1: Name, Col 2: Acronym, Col 3: Source, Col 4: Rank).
    - Trailing columns (6+) are assumed to be Field of Research (FOR) codes.
    - Year is extracted strictly from the filename.
    """
    try:
        # Try reading with header detection
        df = pd.read_csv(path)
        col_map = _map_core_columns(df.columns)
        
        # If we didn't find at least 'name' and 'rank', maybe it's headerless
        if "name" not in col_map.values() or "rank" not in col_map.values():
            raise ValueError("Headers not found or mapping failed")
            
        df = df.rename(columns=col_map)
    except Exception:
        # Fallback: positional indexing if headers are missing/messy
        df = pd.read_csv(path, header=None, engine="python")
        cols = list(df.columns)
        if len(cols) < 5:
            raise ValueError(f"{path.name}: expected at least 5 columns, got {len(cols)}")
        
        # Mapping by standard CORE export positions
        df = df.rename(columns={
            cols[1]: "name",
            cols[2]: "acronym",
            cols[3]: "source",
            cols[4]: "rank"
        })

    # Find where FOR codes start (usually after 'rank' or 'dblp')
    # If we have 'dblp' at col 5, FOR codes start at 6.
    # We'll just assume everything after rank/dblp columns are FOR codes.
    try:
        rank_idx = df.columns.get_loc("rank")
        # Try to skip 'dblp' if it exists right after rank
        for_start = rank_idx + 1
        if for_start < len(df.columns) and str(df.columns[for_start]).lower() == "dblp":
            for_start += 1
    except Exception:
        for_start = 5 # default fallback

    # Clean FOR columns
    for col in df.columns[for_start:]:
        ser = df[col]
        ser = ser.where(ser.notna(), "")
        ser = ser.astype(str)
        ser = ser.str.replace(r"\.0+$", "", regex=True)
        df[col] = ser.str.strip()

    # build a trimmed frame
    out = pd.DataFrame({
        "name":    df["name"].astype(str).str.strip(),
        "acronym": df["acronym"].astype(str).str.strip(),
        "source":  df["source"].astype(str).str.strip(),
        "rank":    df["rank"].apply(normalize_rank),
    })
    
    # year strictly from filename
    year = infer_year_from_filename(path)
    out["year"] = year

    # FOR codes
    out["for_codes"] = df.apply(lambda r: collect_for_codes(r, for_start), axis=1)

    # matching keys
    out["core_key_name"] = out["name"].apply(norm_key)
    out["core_key_acr"]  = out["acronym"].apply(norm_key)

    return out[["year", "name", "acronym", "rank", "for_codes", "core_key_name", "core_key_acr"]]

def build_core_yearly(input_folder="core_raw", output_dir="out/core", file_list=None):
    input_path = Path(input_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    if file_list:
        files_to_process = [input_path / f for f in file_list]
    else:
        files_to_process = sorted(input_path.glob("*.csv"))

    buckets = {}
    for f in files_to_process:
        try:
            df = load_core_csv(f)
            y = df["year"].iloc[0]
            buckets.setdefault(y, []).append(df)
        except Exception as e:
            logger.warning(f"{f.name}: {e}")

    if not buckets:
        logger.warning("No CORE CSV files parsed.")
        return

    for year, frames in buckets.items():
        big = pd.concat(frames, ignore_index=True)
        big = big.sort_values(["core_key_name", "core_key_acr"])
        big = big.drop_duplicates(["core_key_name", "core_key_acr"], keep="first")

        out = big[["year", "name", "acronym", "rank", "for_codes"]].copy()

        out_path = output_dir / f"core_normalized_{year}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"✅ Wrote {len(out):,} rows → {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from functionalities.common import resolve_path
    build_core_yearly(input_folder=resolve_path("core_raw"), output_dir=resolve_path("out/core"))
