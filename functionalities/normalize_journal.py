import pandas as pd
from pathlib import Path
import csv
import logging
from functionalities.common import (
    SYNONYMS, INDEX_PAT, normalize_text, norm_key, 
    infer_year_from_filename, smart_read_excel
)

logger = logging.getLogger(__name__)

# ---- Constants & Config ----

CANONICAL_ORDER = [
    "category",
    "index",
    "journal_title",
    "issn_print",
    "issn_electronic",
    "score_if",
    "score_ais",
]

AREA_TO_INDEX = {
    "SOCIAL SCIENCES": "SSCI",
    "SOCIAL SCIENCENCES": "SSCI",
    "SCIENCE": "SCIE",
    "ARTS & HUMANITIES": "AHCI",
}

# --- Internal Helpers ---

def _map_columns(df_cols):
    """Map dataframe columns to canonical names using SYNONYMS."""
    mapped = {}
    used = set()
    for col in df_cols:
        coln = normalize_text(col)
        hit = None
        for canon, alts in SYNONYMS.items():
            if canon in used:
                continue
            if coln in alts or any(coln.startswith(a) for a in alts):
                hit = canon
                break
        if hit:
            mapped[col] = hit
            used.add(hit)
    return mapped

def _normalize_quartile(v):
    if pd.isna(v):
        return "N/A"
    s = str(v).strip().upper()
    if s.startswith("Q"):
        return s
    try:
        n = int(float(s))
        return f"Q{n}" if 1 <= n <= 4 else "N/A"
    except ValueError:
        return "N/A"

def _extract_index_from_category(df: pd.DataFrame) -> pd.DataFrame:
    """Detect SCIE/SSCI/AHCI/ESCI only if they appear at the END of the category string."""
    df = df.copy()
    if "index" not in df.columns:
        df["index"] = "N/A"

    token = df["category"].astype(str).str.extract(INDEX_PAT, expand=False)
    has_token = token.notna()
    token = token.str.upper().str.strip()

    df.loc[has_token, "index"] = token[has_token]
    df.loc[has_token, "category"] = (
        df.loc[has_token, "category"]
        .astype(str)
        .str.replace(INDEX_PAT, "", regex=True)
        .str.rstrip(" -–")
        .str.strip()
    )

    df["index"] = df["index"].replace({"": "N/A"})
    return df

def _drop_header_like_rows(df):
    col_norm = {c: normalize_text(c) for c in df.columns}

    def is_header_like(row):
        hits = 0
        for c, v in row.items():
            if pd.isna(v):
                continue
            vn = normalize_text(v)
            if vn == col_norm[c]:
                hits += 1
        jt = row.get("journal_title", None)
        if isinstance(jt, str):
            jn = normalize_text(jt)
            if "revista" in jn and "revistele marcate" in jn:
                return True
        return hits >= 2

    return df[~df.apply(is_header_like, axis=1)].copy()

def _clean_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace and normalize scores/ISSNs."""
    df = df.copy()
    for c in df.columns:
        if c == "journal_title":
            df[c] = df[c].astype(str).str.replace(r"\[\*\]", "", regex=True).str.strip()
        else:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Forward-fill merged cells
    for c in ("category", "index"):
        if c in df.columns:
            df[c] = df[c].replace("", pd.NA).ffill()

    if "score_if" in df.columns:
        df["score_if"] = df["score_if"].apply(_normalize_quartile)
    if "score_ais" in df.columns:
        df["score_ais"] = df["score_ais"].apply(_normalize_quartile)

    for col in ["issn_print", "issn_electronic"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: "N/A" if pd.isna(x) or str(x).strip() in ("", "nan", "None") else str(x).strip()
            )
    return df

# --- Main Processor ---

def normalize_one_file(path, sheet_name=0):
    """
    Reads and normalizes a single journal file from Romanian ministerial lists.
    
    Format Assumptions:
    - Files are typically converted from PDF to Excel.
    - Columns may have Romanian headers (e.g., 'Revista', 'Zona').
    - 'score' mapping depends on filename: 'ais' in filename maps to AIS scores, 
      otherwise defaults to JIF (Impact Factor) scores.
    - Categories often contain the index (SCIE, SSCI, etc.) at the end of the string.
    """
    df = smart_read_excel(path, sheet_name=sheet_name)
    if df.empty:
        raise ValueError("Empty sheet")

    # 1. Rename columns
    col_map = _map_columns(df.columns)
    df = df.rename(columns=col_map)

    # 2. Contextual renaming for 'score'
    filename = str(path).lower()
    if "score_if" not in df.columns and "score_ais" not in df.columns and "score" in df.columns:
        if "ais" in filename:
            df = df.rename(columns={"score": "score_ais"})
        else:
            df = df.rename(columns={"score": "score_if"})

    # 3. Ensure required columns exist
    for col in ["category", "index", "journal_title", "issn_print", "issn_electronic", "score_if","score_ais", "top"]:
        if col not in df.columns:
            df[col] = "N/A"

    # 4. Extract and Map metadata
    df = _extract_index_from_category(df)
    df["index"] = df["index"].astype(str).str.strip().replace(AREA_TO_INDEX).str.upper()
    df = _drop_header_like_rows(df)

    # 5. Filter and Clean
    keep = [c for c in CANONICAL_ORDER if c in df.columns]
    if not keep:
        raise ValueError("No recognizable columns found")
    df = df[keep]
    df = _clean_data_types(df)

    # Final ordering
    order = [c for c in ["category", "index", "journal_title", "issn_print", "issn_electronic", "score_if","score_ais", "top"] if c in df.columns]
    return df[order]

def _coalesce(values):
    for v in values:
        if pd.notna(v) and str(v).strip().upper() != "N/A":
            return v
    return "N/A"

def build_yearly_outputs(input_folder="journal_raw", output_dir="out/journal", log_file="log.csv", sheet_name=0, file_list=None):
    input_path = Path(input_folder)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    buckets = {}

    files_to_process = []
    if file_list:
        files_to_process = [input_path / f for f in file_list]
    else:
        files_to_process = sorted(input_path.glob("*.xls*"))

    for f in files_to_process:
        year = infer_year_from_filename(f.name)
        if year == "N/A":
            logger.warning(f"Skipping {f.name}: no year in filename")
            continue
        try:
            df = normalize_one_file(f, sheet_name=sheet_name)
            df["year"] = year
            buckets.setdefault(year, []).append(df)
        except Exception as e:
            logger.warning(f"{f.name}: {e}")

    if not buckets:
        logger.warning("No journal files parsed.")
        return

    with open(log_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "rows", "output_file"])

        for year, frames in buckets.items():
            big = pd.concat(frames, ignore_index=True)
            big = big.copy()
            
            # Key generation for grouping
            big["cat_key"] = big["category"].apply(norm_key)
            big["jtitle_key"] = big["journal_title"].apply(norm_key)
            big["issn_p_key"] = big["issn_print"].apply(lambda x: norm_key(str(x)).replace("-", ""))
            big["issn_e_key"] = big["issn_electronic"].apply(lambda x: norm_key(str(x)).replace("-", ""))
            big["idx_key"] = big["index"].apply(norm_key)

            keys = ["issn_p_key", "issn_e_key", "cat_key", "idx_key", "jtitle_key"]
            for k in keys:
                if k not in big.columns:
                    big[k] = "N/A"

            agg = {"score_if": _coalesce, "score_ais": _coalesce}
            for c in ["category", "index", "journal_title", "issn_print", "issn_electronic"]:
                agg.setdefault(c, "first")

            grouped = big.groupby(keys, dropna=False, as_index=False).agg(agg)

            cols = ["category", "index", "journal_title", "issn_print", "issn_electronic", "score_if", "score_ais"]
            for c in cols:
                if c not in grouped.columns:
                    grouped[c] = "N/A"
            grouped = grouped[cols]
            
            out_path = Path(output_dir) / f"normalized_{year}.xlsx"
            grouped.to_excel(out_path, index=False)

            writer.writerow([year, len(grouped), str(out_path)])
            logger.info(f"✅ Wrote {len(grouped):,} rows → {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from functionalities.common import resolve_path
    build_yearly_outputs(input_folder=resolve_path("journal_raw"), output_dir=resolve_path("out/journal"), sheet_name=0)
