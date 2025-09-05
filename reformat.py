#!/usr/bin/env python3
"""
combine_biblio.py
-----------------
Combine Scopus CSV and Web of Science (WoS XLS/XLSX) exports that share
the same basename into a single formatted Excel file.

- Rows are matched by DOI (preferred) or Title.
- If values differ, they are joined with " / ".
- If a row exists in only one file, it is included as-is.

Output columns:
- Nr.crt.
- Titlu
- Autori
- FORUM (Revista, Conferința)   (smart title case)
- Volum, nr., pg.   (Volume/Issue/Pages OR Article No. + DOI link)
- An
- Adresă URL        (https://doi.org/<DOI>)
- Categorie IF
- Categorie AIS

Usage:
    python combine_biblio.py --input exports --output out

Requires:
    pip install pandas openpyxl xlrd
"""
import argparse
import os
import pandas as pd
import re
import glob


# ---------- Utilities ----------

def to_text(x):
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()


def smart_title_case(text: str) -> str:
    if not text:
        return ""
    small_words = {"and", "or", "of", "the", "in", "on", "at", "for", "with", "a", "an"}
    words = text.split()
    result = []
    for i, word in enumerate(words):
        lw = word.lower()
        if i not in (0, len(words) - 1) and lw in small_words:
            result.append(lw)
        else:
            result.append(lw.capitalize())
    return " ".join(result)


def doi_to_url(doi: str) -> str:
    if not doi:
        return ""
    # Support multiple DOIs joined by " / "
    parts = [f"https://doi.org/{to_text(d)}" for d in doi.split(" / ") if to_text(d)]
    return " / ".join(parts)


def normalize_value(v: str) -> str:
    """Normalize text for deduplication."""
    v = v.lower()
    v = v.replace("&", "and")
    v = re.sub(r"[^a-z0-9]+", " ", v)  # replace non-alphanumeric with space
    v = re.sub(r"\s+", " ", v).strip()
    return v


def normalize_title(v: str) -> str:
    if pd.isna(v): return ""
    v = str(v).lower().replace("&", "and")
    v = re.sub(r"[^a-z0-9]+", " ", v)
    return re.sub(r"\s+", " ", v).strip()

# ---------- helpers for picking "best" scores ----------

def _best_quartile(vals):
    """
    Choose best among quartile-like labels (Q1..Q4, 'N/A', empty).
    Highest priority: Q1 > Q2 > Q3 > Q4 > anything else.
    """
    order = {"q1": 4, "q2": 3, "q3": 2, "q4": 1}
    best = None
    best_rank = -1
    for v in vals:
        if not v:
            continue
        k = str(v).strip().lower()
        rank = order.get(k, 0)
        if rank > best_rank:
            best_rank = rank
            best = v
    return best or ""

def _best_numeric(vals):
    """
    Choose largest numeric value from a list of strings; ignore non-numerics.
    """
    best_val = None
    best_raw = ""
    for v in vals:
        try:
            x = float(str(v).replace(",", "."))
        except Exception:
            continue
        if best_val is None or x > best_val:
            best_val = x
            best_raw = v
    return best_raw or ""

def _best_if_or_quartile(vals):
    """
    IF may be provided as quartile (Q1..Q4) or numeric. Prefer:
      - best quartile if any quartiles exist
      - otherwise, the largest numeric
    """
    vals = [to_text(v) for v in vals if to_text(v)]
    if not vals:
        return ""
    any_quart = any(str(v).strip().upper().startswith("Q") for v in vals)
    return _best_quartile(vals) if any_quart else _best_numeric(vals)

# ---------- year-fallback lookup (use latest <= paper year; if none, use latest available) ----------

def _lookup_journal_best(jidx, norm, year):
    """
    Use ONLY the exact year if it exists for (norm, year).
    If not, fall back to the latest year strictly < given year.
    If still none, return empty.
    """
    sub = jidx[jidx["norm_title"] == norm]
    if sub.empty or pd.isna(year):
        return "", ""
    y = int(year)

    exact = sub[sub["year"] == y]
    if not exact.empty:
        chosen = exact.iloc[0]  # already aggregated to "best" per (title, year)
        return to_text(chosen.get("if", "")), to_text(chosen.get("ais", ""))

    # fallback: latest prior year only
    prior = sub[sub["year"] < y]
    if prior.empty:
        return "", ""
    chosen = prior.sort_values("year", ascending=False).iloc[0]
    return to_text(chosen.get("if", "")), to_text(chosen.get("ais", ""))


def _lookup_core_best(cidx, norm, year):
    """
    Use ONLY the exact year if available; otherwise fall back to latest prior year.
    """
    sub = cidx[cidx["norm_title"] == norm]
    if sub.empty or pd.isna(year):
        return ""
    y = int(year)

    exact = sub[sub["year"] == y]
    if not exact.empty:
        return to_text(exact.iloc[0].get("core_rank", ""))

    prior = sub[sub["year"] < y]
    if prior.empty:
        return ""
    return to_text(prior.sort_values("year", ascending=False).iloc[0].get("core_rank", ""))


# ---------- build indexes (aggregate duplicates per (title, year)) ----------

def build_journal_index(journal_dir: str):
    frames = []
    for path in glob.glob(os.path.join(journal_dir, "normalized_*.xlsx")):
        year = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
        df = pd.read_excel(path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        if "journal_title" not in df.columns:
            continue
        sub = pd.DataFrame({
            "norm_title": df["journal_title"].map(normalize_title),
            "year": year,
            "if": df["score_if"] if "score_if" in df.columns else "",
            "ais": df["score_ais"] if "score_ais" in df.columns else "",
        })
        frames.append(sub)

    jidx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["norm_title","year","if","ais"])

    # collapse duplicates per (norm_title, year) keeping BEST scores
    def agg_best(g):
        return pd.Series({
            "if": _best_if_or_quartile(g["if"].tolist()),
            "ais": _best_quartile(g["ais"].tolist())
        })

    if not jidx.empty:
        jidx = jidx.groupby(["norm_title", "year"], as_index=False).apply(
            agg_best, include_groups=False
        )
        # ensure year is int
        jidx["year"] = jidx["year"].astype(int)
    return jidx


def build_core_index(core_dir: str):
    frames = []
    for path in glob.glob(os.path.join(core_dir, "core_normalized_*.csv")):
        year = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        if "name" not in df.columns or "rank" not in df.columns:
            continue
        sub = pd.DataFrame({
            "norm_title": df["name"].map(normalize_title),
            "year": year,
            "core_rank": df["rank"],
        })
        frames.append(sub)

    cidx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["norm_title","year","core_rank"])
    if not cidx.empty:
        # if multiple ranks exist for same (title, year), keep the best (A* > A > B > C > else)
        order = {"a*": 4, "a": 3, "b": 2, "c": 1}
        def best_core(g):
            best = ""
            best_rank = -1
            for v in g["core_rank"].tolist():
                k = str(v).strip().lower()
                r = order.get(k, 0)
                if r > best_rank:
                    best_rank, best = r, v
            return pd.Series({"core_rank": best})

        cidx = cidx.groupby(["norm_title", "year"], as_index=False).apply(
            best_core, include_groups=False
        )
        cidx["year"] = cidx["year"].astype(int)
    return cidx


# ---------- lookup that splits FORUM by '/' and applies the fallbacks ----------

def lookup_scores(title_raw, year, jidx, cidx):
    """Lookup IF/AIS/CORE for possibly multiple forum names separated by '/', with year fallback and 'best' picks."""
    if not title_raw:
        return "", "", ""
    parts = [t.strip() for t in str(title_raw).split("/") if t.strip()]

    scores_if, scores_ais, scores_core = [], [], []
    for part in parts:
        norm = normalize_title(part)
        j_if, j_ais = _lookup_journal_best(jidx, norm, year)
        c_rank = _lookup_core_best(cidx, norm, year)
        if j_if:  scores_if.append(j_if)
        if j_ais: scores_ais.append(j_ais)
        if c_rank: scores_core.append(c_rank)

    return " / ".join(scores_if), " / ".join(scores_ais), " / ".join(scores_core)


def attach_scores(formatted_df, journal_dir="out/journal", core_dir="out/core"):
    jidx = build_journal_index(journal_dir)
    cidx = build_core_index(core_dir)

    formatted_df = formatted_df.copy()
    formatted_df["year"] = pd.to_numeric(formatted_df["An"], errors="coerce")

    results = formatted_df.apply(
        lambda r: lookup_scores(r["FORUM (Revista, Conferința)"], r["year"], jidx, cidx),
        axis=1, result_type="expand"
    )
    results.columns = ["Categorie IF", "Categorie AIS", "CORE Rank"]

    df = pd.concat([formatted_df, results], axis=1)

    # remove duplicates
    df = df.drop_duplicates(
        subset=["Titlu", "Autori", "FORUM (Revista, Conferința)", "An"],
        keep="first"
    ).reset_index(drop=True)

    # drop helper cols if you don’t want them
    if "norm_title" in df.columns:
        df = df.drop(columns=["norm_title"])
    return df



def merge_possible(df_row, candidates, clean_authors=False):
    """Return merged values from all matching columns, joined with / if different."""
    vals = []
    seen = set()
    for c in candidates:
        if c in df_row.index and "id" not in c.lower():   # skip IDs
            v = to_text(df_row[c])
            if v:
                if clean_authors:
                    # remove anything in parentheses
                    v = re.sub(r"\s*\([^)]*\)", "", v).strip()
                norm = normalize_value(v)
                if norm not in seen:
                    seen.add(norm)
                    vals.append(v.strip())
    return " / ".join(vals)



def compose_vol_issue_pages(row):
    parts = []

    vol = merge_possible(row, ["Volume", "VL"])
    iss = merge_possible(row, ["Issue", "IS"])
    pstart = merge_possible(row, ["Page start", "Start Page", "BP", "SP"])
    pend = merge_possible(row, ["Page end", "End Page", "EP"])
    artno = merge_possible(row, ["Art. No.", "AR", "Article Number"])

    if vol:
        parts.append(f"Vol {vol}")
    if iss:
        parts.append(f"No {iss}")
    if pstart or pend:
        if pstart and pend:
            parts.append(f"pp {pstart}-{pend}")
        else:
            parts.append(f"pp {pstart or pend}")
    elif artno:
        parts.append(f"Art. No. {artno}")

    base = ", ".join(parts)

    doi_val = merge_possible(row, ["DOI", "DI"])
    doi_url = doi_to_url(doi_val)

    if base and doi_url:
        return f"{base} | {doi_url}"
    return doi_url or base


def merge_rows(srow, wrow):
    merged = {}
    for col in set(srow.index) | set(wrow.index):
        v1 = to_text(srow.get(col, ""))
        v2 = to_text(wrow.get(col, ""))
        if v1 and v2:
            merged[col] = v1 if v1 == v2 else f"{v1} / {v2}"
        else:
            merged[col] = v1 or v2
    return merged


def combine(scopus_df, wos_df):
    merged_rows = []
    used_wos = set()

    for _, srow in scopus_df.iterrows():
        sdoi = to_text(srow.get("DOI", "")) or to_text(srow.get("DI", ""))
        smatch = None
        if sdoi:
            candidates = wos_df[wos_df.get("DOI", wos_df.get("DI", "")).fillna("").str.lower() == sdoi.lower()]
            if not candidates.empty:
                smatch = candidates.iloc[0]
        else:
            stitle = to_text(srow.get("Title", srow.get("Article Title", ""))).lower()
            if stitle:
                candidates = wos_df[
                    wos_df.get("Title", wos_df.get("Article Title", "")).fillna("").str.lower() == stitle
                ]
                if not candidates.empty:
                    smatch = candidates.iloc[0]

        if smatch is not None:
            merged_rows.append(merge_rows(srow, smatch))
            used_wos.add(smatch.name)
        else:
            merged_rows.append(srow.to_dict())

    for idx, wrow in wos_df.iterrows():
        if idx not in used_wos:
            merged_rows.append(wrow.to_dict())

    return pd.DataFrame(merged_rows)

def read_any_table(path: str) -> pd.DataFrame:
    """
    Robust reader for Scopus/WoS exports.

    - .csv/.tsv/.txt: try common separators & encodings
    - .xls/.xlsx: try Excel engines; if that fails, try tab-delimited text (many WoS .xls are TSV)
    """
    ext = os.path.splitext(path)[1].lower()

    # Text-like formats
    if ext in {".csv", ".tsv", ".txt"}:
        # Try straightforward CSV
        try:
            df = pd.read_csv(path, dtype=str)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        # Try alternative seps/encodings
        for sep in [",", "\t", ";", "|"]:
            for enc in ["utf-8", "utf-8-sig", "latin1", "utf-16", "utf-16le"]:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
        raise RuntimeError(f"Unable to parse text file: {path}")

    # Excel formats
    if ext in {".xls", ".xlsx"}:
        # Try native Excel readers
        for engine in [None, "openpyxl", "xlrd"]:
            try:
                df = pd.read_excel(path, engine=engine, dtype=str)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        # Fallback: mislabeled WoS .xls as TSV
        for enc in ["utf-16", "utf-16le", "utf-8-sig", "latin1", "utf-8"]:
            try:
                df = pd.read_csv(path, sep="\t", encoding=enc, dtype=str)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        raise RuntimeError(f"Unable to read Excel file: {path}. Try installing 'openpyxl' and 'xlrd'.")

    # Last resort: let pandas guess
    return pd.read_csv(path, dtype=str)

def format_final(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Nr.crt."] = range(1, len(df) + 1)

    out["Titlu"] = df.apply(lambda r: merge_possible(r, ["Title", "TI", "Article Title"]), axis=1)
    out["Autori"] = df.apply(
        lambda r: merge_possible(r, ["Author Full Names", "Author full names", "AF", "AU"], clean_authors=True),
        axis=1
    )

    out["FORUM (Revista, Conferința)"] = df.apply(
        lambda r: " / ".join(
            smart_title_case(x)
            for x in merge_possible(r, ["Source title", "Source Title", "SO", "Publication Name"]).split(" / ")
        ),
        axis=1
    )

    out["Volum, nr., pg."] = df.apply(compose_vol_issue_pages, axis=1)
    out["An"] = df.apply(lambda r: merge_possible(r, ["Year", "PY", "Publication Year"]), axis=1)

    out["Adresă URL"] = df.apply(
        lambda r: " / ".join(
            doi_to_url(x)
            for x in merge_possible(r, ["DOI", "DI"]).split(" / ")
            if x
        ),
        axis=1
    )
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input folder with paired Scopus/WoS files")
    ap.add_argument("--output", "-o", required=True, help="Output folder for merged Excel files")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # group files by basename
    files = os.listdir(args.input)
    basenames = {}
    for f in files:
        base, ext = os.path.splitext(f)
        basenames.setdefault(base, []).append(os.path.join(args.input, f))

    for base, paths in basenames.items():
        scopus_path = next((p for p in paths if p.lower().endswith(".csv")), None)
        wos_path = next((p for p in paths if p.lower().endswith((".xls", ".xlsx"))), None)

        if scopus_path and wos_path:
            # both exports available → combine, then format
            scopus_df = pd.read_csv(scopus_path, dtype=str)
            wos_df = pd.read_excel(wos_path, dtype=str)
            raw = combine(scopus_df, wos_df)
            suffix = "_merged"
        elif scopus_path or wos_path:
            # only one export available → just use it as-is, then format
            only_path = scopus_path or wos_path
            raw = read_any_table(only_path)
            suffix = "_single"
        else:
            # nothing usable in this group
            continue

        # format + attach scores (IF/AIS/CORE) as usual
        formatted = format_final(raw)
        formatted = attach_scores(formatted, journal_dir="out/journal", core_dir="out/core")

        out_path = os.path.join(args.output, f"{base}{suffix}.xlsx")
        formatted.to_excel(out_path, index=False)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
