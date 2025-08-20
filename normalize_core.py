import pandas as pd
from pathlib import Path
import re

# ---------- helpers ----------
YEAR_RE = re.compile(r"(19|20)\d{2}")

def infer_year_from_filename(pathlike) -> str:
    m = YEAR_RE.search(Path(pathlike).name)
    return m.group(0) if m else "N/A"

def norm_key(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())

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
def load_core_csv(path: Path) -> pd.DataFrame:
    """
    Expected layout:
    Col0=id(ignored), Col1=name, Col2=acronym, Col3=source (e.g., CORE2021),
    Col4=rank, Col5=dblp(Yes/No), Col6+ = FOR codes.
    If headers exist/are messy we still map by position.
    """
    try:
        df = pd.read_csv(path)
        # if someone exported with headers, keep them but we’ll still index by position
    except Exception:
        df = pd.read_csv(path, header=None, engine="python")

    # force positional rename when possible

    cols = list(df.columns)
    if len(cols) < 5:
        raise ValueError(f"{path.name}: expected at least 5 columns, got {len(cols)}")

    # safe positional picks
    name_col   = cols[1]
    acr_col    = cols[2]
    src_col    = cols[3]
    rank_col   = cols[4]
    for_start  = 6 if len(cols) >= 7 else max(cols.index(rank_col) + 1, 5)

    # 🔧 Clean FOR columns (from col index 6 onward): keep as strings, drop .0, remove 'nan'
    for_start = 6 if len(df.columns) >= 7 else max(len(df.columns) - 1, 5)
    for col in df.columns[for_start:]:
        ser = df[col]
        ser = ser.where(ser.notna(), "")
        ser = ser.astype(str)
        # strip trailing .0 / .00 etc
        ser = ser.str.replace(r"\.0+$", "", regex=True)
        df[col] = ser.str.strip()

    # build a trimmed frame
    out = pd.DataFrame({
        "name":    df[name_col].astype(str).str.strip(),
        "acronym": df[acr_col].astype(str).str.strip(),
        "source":  df[src_col].astype(str).str.strip(),
        "rank":    df[rank_col].apply(normalize_rank),
    })
    # year strictly from filename (your requirement)
    year = infer_year_from_filename(path)
    out["year"] = year

    # FOR codes from remaining columns
    out["for_codes"] = df.apply(lambda r: collect_for_codes(r, for_start), axis=1)

    # matching keys (for within-year de-dup)
    out["core_key_name"] = out["name"].apply(norm_key)
    out["core_key_acr"]  = out["acronym"].apply(norm_key)

    # final order
    return out[["year", "name", "acronym", "rank", "for_codes", "core_key_name", "core_key_acr"]]

# ---------- batch: one normalized CSV per year ----------
def build_core_yearly(input_folder="core_raw", output_dir="core_out"):
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # bucket by year from filename
    buckets = {}
    for f in sorted(input_folder.glob("*.csv")):
        try:
            df = load_core_csv(f)
            y = df["year"].iloc[0]
            buckets.setdefault(y, []).append(df)
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")

    if not buckets:
        raise RuntimeError("No CORE CSV files parsed.")

    # write one CSV per year; de-dup within year by (name+acronym) key
    for year, frames in buckets.items():
        big = pd.concat(frames, ignore_index=True)
        big = big.sort_values(["core_key_name", "core_key_acr"])
        big = big.drop_duplicates(["core_key_name", "core_key_acr"], keep="first")

        # keep tidy columns for output
        out = big[["year", "name", "acronym", "rank", "for_codes"]].copy()

        out_path = output_dir / f"core_normalized_{year}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Wrote {len(out):,} rows → {out_path}")

if __name__ == "__main__":
    # change folders as needed
    build_core_yearly(input_folder="core_raw", output_dir="out/core")
