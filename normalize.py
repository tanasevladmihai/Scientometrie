import pandas as pd
from pathlib import Path
import unicodedata
import re
import csv

# ---- Config: synonyms for column names (case/diacritics insensitive) ----
SYNONYMS = {
    "category": {
        "web of science category", "wos category", "category", "categorie", "subject category", "subdomeniu/web of science category-index", "subdomeniu/web of science category"
    },
    "index": {
        "index", "indice", "indexare", "wos index", "edition"
    },
    "journal_title": {
        "revista", "journal", "journal title", "journal name", "title", "publication title", "full journal title", "denumirea revistei"
    },
    "issn_print": {
        "issn", "p-issn", "print issn"
    },
    "issn_electronic": {
        "eissn", "e-issn", "electronic issn"
    },
    "score_if": {
        "jif quartile", "if quartile", "jif", "journal impact factor quartile", "impact factor quartile", "q jif", "zona/q jif"
    },
    "score_ais": {
        "ais quartile", "ais", "q ais"
    },
    "score": {
        "zona", "quartile"
    },
    "top": {
        "top", "rank", "rank in category", "percentile", "top percent", "top%", "loc"
    },
}

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
    "SOCIAL SCIENCENCES": "SSCI",  # typo fix -> SSCI
    "SCIENCE": "SCIE",
    "ARTS & HUMANITIES": "AHCI",
}

# --- pull SCIE/SSCI/AHCI/ESCI from the end of Category ---

INDEX_PAT = re.compile(
    r"\s*[-–]?\s*\(?\s*(SCIE|SSCI|AHCI|ESCI)\s*\)?\s*$",
    re.IGNORECASE,
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    s = _strip_accents(s).lower()
    s = " ".join(s.split())
    return s

def _map_columns(df_cols):
    """Map dataframe columns to canonical names using SYNONYMS."""
    mapped = {}
    used = set()
    for j, col in enumerate(df_cols):
        coln = _norm(col)
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


def extract_index_from_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect SCIE/SSCI/AHCI/ESCI only if they appear at the END
    of the category string (with optional dashes/parentheses).
    """
    df = df.copy()
    if "index" not in df.columns:
        df["index"] = "N/A"

    # Extract token only if it's at the end (safe against "SCIENCE")
    token = df["category"].astype(str).str.extract(INDEX_PAT, expand=False)

    has_token = token.notna()
    token = token.str.upper().str.strip()

    # Fill index
    df.loc[has_token, "index"] = token[has_token]

    # Strip the trailing index part from category
    df.loc[has_token, "category"] = (
        df.loc[has_token, "category"]
        .astype(str)
        .str.replace(INDEX_PAT, "", regex=True)
        .str.rstrip(" -–")
        .str.strip()
    )

    df["index"] = df["index"].replace({"": "N/A"})
    return df



def drop_header_like_rows(df):
    # map column -> normalized column name
    col_norm = {c: _norm(c) for c in df.columns}

    def is_header_like(row):
        hits = 0
        for c, v in row.items():
            if pd.isna(v):
                continue
            vn = _norm(v)
            # same as its own column name (e.g., "ISSN", "Top", "Index")
            if vn == col_norm[c]:
                hits += 1
        # Special case
        jt = row.get("journal_title", None)
        if isinstance(jt, str):
            jn = _norm(jt)
            if "revista" in jn and "revistele marcate" in jn:
                return True
        # if 2+ cells look like column labels, treat as header row
        return hits >= 2

    return df[~df.apply(is_header_like, axis=1)].copy()

def normalize_one_file(path, sheet_name=0):
    """
    Read Excel trusting the FIRST ROW as header. No repeated-header detection.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    if df.empty:
        raise ValueError("Empty sheet")

    # Rename columns by synonyms
    col_map = _map_columns(df.columns)
    df = df.rename(columns=col_map)

    filename = str(path).lower()

    if "score_if" not in df.columns and "score_ais" not in df.columns and "score" in df.columns:
        if "ais" in filename:
            df = df.rename(columns={"score": "score_ais"})
        elif "jif" in filename or "if" in filename:
            df = df.rename(columns={"score": "score_if"})
        else:
            # fallback: assume IF
            df = df.rename(columns={"score": "score_if"})

    if "score_if" not in df.columns:
        df["score_if"] = "N/A"
    if "score_ais" not in df.columns:
        df["score_ais"] = "N/A"

    for col in ["category", "index", "journal_title", "issn_print", "issn_electronic", "score_if","score_ais", "top"]:
        if col not in df.columns:
            df[col] = "N/A"

    df = extract_index_from_category(df)
    df["index"] = (
        df["index"]
        .astype(str).str.strip()
        .replace(AREA_TO_INDEX)  # map areas -> index
        .str.upper()
    )
    df = drop_header_like_rows(df)

    # Keep only known columns if present
    keep = [c for c in CANONICAL_ORDER if c in df.columns]
    if not keep:
        raise ValueError("No recognizable columns found")
    df = df[keep].copy()

    # Clean whitespace
    for c in df.columns:
        if c == "journal_title":
            df[c] = df[c].astype(str).str.replace(r"\[\*\]", "", regex=True).str.strip()
        else:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Forward-fill likely merged category/index gaps (optional but harmless)
    for c in ("category", "index"):
        if c in df.columns:
            df[c] = df[c].replace("", pd.NA).ffill()

    # Normalize score (Q1->1, "1.0"->1)
    if "score" in df.columns:
        df["score"] = df["score"].apply(_normalize_quartile)

    df["score_if"] = df["score_if"].apply(_normalize_quartile)
    df["score_ais"] = df["score_ais"].apply(_normalize_quartile)

    for col in ["issn_print", "issn_electronic"]:
        if col not in df.columns:
            df[col] = "N/A"
        else:
            df[col] = df[col].apply(
                lambda x: "N/A" if pd.isna(x) or str(x).strip() in ("", "nan", "None") else str(x).strip()
            )

    # ISSN helpers
    # if "issn" in df.columns:
    #     df["issn_raw"] = df["issn"]
    #     df["issn_norm"] = (
    #         df["issn"].astype(str)
    #         .str.replace(r"[^0-9Xx]", "", regex=True)
    #         .str.upper()
    #     )

    # Basic de-dup
    # subset = [c for c in ["journal_title", "issn_norm", "category"] if c in df.columns]
    # if subset:
    #     df = df.drop_duplicates(subset=subset, keep="first")

    # Order columns nicely
    order = [c for c in ["category", "index", "journal_title", "issn_print", "issn_electronic", "score_if","score_ais", "top"] if c in df.columns]
    return df[order]

def build_unified_db(input_folder="db_raw", output_file="db_normalized.xlsx", sheet_name=0):
    frames = []
    for xlsx in Path(input_folder).glob("*.xls*"):  # matches .xls and .xlsx
        try:
            df = normalize_one_file(xlsx, sheet_name=sheet_name)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] {xlsx.name}: {e}")
    if not frames:
        raise RuntimeError("No files parsed.")
    out = pd.concat(frames, ignore_index=True)
    out.to_excel(output_file, index=False)
    print(f"✅ Wrote {len(out):,} rows to {output_file}")
    return out

YEAR_RE = re.compile(r"(19|20)\d{2}")

def _infer_year_from_filename(pathlike):
    name = Path(pathlike).stem
    m = YEAR_RE.search(name)
    return m.group(0) if m else None

def _coalesce(values):
    # first non-"N/A"
    for v in values:
        if pd.notna(v) and str(v).strip().upper() != "N/A":
            return v
    return "N/A"

def _norm_key(s: str) -> str:
    if pd.isna(s):
        return "N/A"
    s = str(s).upper().strip()
    s = re.sub(r"\s+", "", s)   # remove ALL spaces
    return s

def build_yearly_outputs(input_folder="db_raw", output_dir="out", log_file="log.csv", sheet_name=0):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    buckets = {}  # year -> list[df]

    for f in Path(input_folder).glob("*.xls*"):
        year = _infer_year_from_filename(f.name)
        if not year:
            print(f"[WARN] Skipping {f.name}: no year in filename")
            continue
        try:
            df = normalize_one_file(f, sheet_name=sheet_name)
            df["year"] = year
            buckets.setdefault(year, []).append(df)
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")

    if not buckets:
        raise RuntimeError("No files parsed for any year.")

    # open CSV log once
    with open(log_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", "rows", "output_file"])  # header

        for year, frames in buckets.items():
            big = pd.concat(frames, ignore_index=True)

            # ---- your grouping logic stays here ----
            big = big.copy()
            big["cat_key"] = big["category"].apply(_norm_key)
            big["jtitle_key"] = big["journal_title"].apply(_norm_key)
            big["issn_p_key"] = big["issn_print"].apply(lambda x: _norm_key(str(x)).replace("-", ""))
            big["issn_e_key"] = big["issn_electronic"].apply(lambda x: _norm_key(str(x)).replace("-", ""))
            big["idx_key"] = big["index"].apply(_norm_key)

            keys = ["issn_p_key", "issn_e_key", "cat_key", "idx_key", "jtitle_key"]
            for k in keys:
                if k not in big.columns:
                    big[k] = "N/A"

            agg = {
                "score_if": _coalesce,
                "score_ais": _coalesce,
            }
            for c in ["category", "index", "journal_title", "issn_print", "issn_electronic"]:
                agg.setdefault(c, "first")

            grouped = big.groupby(keys, dropna=False, as_index=False).agg(agg)

            cols = ["category", "index", "journal_title",
                    "issn_print", "issn_electronic", "score_if", "score_ais"]
            for c in cols:
                if c not in grouped.columns:
                    grouped[c] = "N/A"
            grouped = grouped[cols]
            out_path = Path(output_dir) / f"normalized_{year}.xlsx"
            grouped.to_excel(out_path, index=False)

            # ---- write to log ----
            writer.writerow([year, len(grouped), str(out_path)])
            print(f"✅ Wrote {len(grouped):,} rows → {out_path}")

if __name__ == "__main__":
    # Adjust paths if needed
    build_yearly_outputs(input_folder="db_raw", output_dir="out", sheet_name=0)
