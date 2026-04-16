import re
import pandas as pd
import unicodedata
from pathlib import Path

# ---------- Constants ----------

YEAR_RE = re.compile(r"(19|20)\d{2}")

INDEX_PAT = re.compile(
    r"\s*[-–]?\s*\(?\s*(SCIE|SSCI|AHCI|ESCI)\s*\)?\s*$",
    re.IGNORECASE,
)

# Synonyms for journal column names
SYNONYMS = {
    "category": {
        "web of science category", "wos category", "category", "categorie", "subject category", 
        "subdomeniu/web of science category-index", "subdomeniu/web of science category"
    },
    "index": {
        "index", "indice", "indexare", "wos index", "edition"
    },
    "journal_title": {
        "revista", "journal", "journal title", "journal name", "title", "publication title", 
        "full journal title", "denumirea revistei"
    },
    "issn_print": {
        "issn", "p-issn", "print issn"
    },
    "issn_electronic": {
        "eissn", "e-issn", "electronic issn"
    },
    "score_if": {
        "jif quartile", "if quartile", "jif", "journal impact factor quartile", 
        "impact factor quartile", "q jif", "zona/q jif"
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

# ---------- Path Utilities ----------

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent

def resolve_path(relative_path: str) -> Path:
    """Resolves a path relative to the project root."""
    return get_project_root() / relative_path

# ---------- Text & Normalization Utilities ----------

def to_text(x) -> str:
    """Converts a value to a stripped string, handling NaNs."""
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()

def strip_accents(s: str) -> str:
    """Removes diacritics from a string."""
    if not s: return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s) -> str:
    """Lowercases, strips accents, and collapses whitespace."""
    s = to_text(s)
    if not s: return ""
    s = strip_accents(s).lower()
    return " ".join(s.split())

def norm_key(s: str) -> str:
    """Strict normalization: uppercase and removes all non-alphanumeric characters."""
    s = to_text(s)
    return re.sub(r"[^A-Z0-9]", "", s.upper())

def normalize_title(v: str) -> str:
    """Normalizes a title for deduplication (lowercase, & -> and, alphanumeric only)."""
    if pd.isna(v): return ""
    v = str(v).lower().replace("&", "and")
    v = re.sub(r"[^a-z0-9]+", " ", v)
    return re.sub(r"\s+", " ", v).strip()

def infer_year_from_filename(pathlike) -> str:
    """Extracts a 4-digit year from a filename."""
    name = Path(pathlike).name
    m = YEAR_RE.search(name)
    return m.group(0) if m else "N/A"

# ---------- Excel Utilities (Phase 2 preview) ----------

def smart_read_excel(path: Path, **kwargs) -> pd.DataFrame:
    """
    Robust reader for Excel files with fallbacks for TSVs mislabeled as .xls.
    """
    path = Path(path)
    # Try native Excel readers
    for engine in [None, "openpyxl", "xlrd"]:
        try:
            df = pd.read_excel(path, engine=engine, **kwargs)
            if not df.empty and df.shape[1] > 1:
                return df
        except Exception:
            continue
            
    # Fallback: mislabeled WoS .xls as TSV
    for enc in ["utf-16", "utf-16le", "utf-8-sig", "latin1", "utf-8"]:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc, dtype=str)
            if not df.empty and df.shape[1] > 1:
                return df
        except Exception:
            continue
            
    raise RuntimeError(f"Unable to read file: {path.name}")
