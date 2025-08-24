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

def merge_possible(df_row, candidates):
    """Return merged values from all matching columns, joined with / if different."""
    vals = []
    seen = set()
    for c in candidates:
        if c in df_row.index:
            v = to_text(df_row[c])
            if v:
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


def format_final(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Nr.crt."] = range(1, len(df) + 1)

    out["Titlu"] = df.apply(lambda r: merge_possible(r, ["Title", "TI", "Article Title"]), axis=1)
    out["Autori"] = df.apply(lambda r: merge_possible(r, ["Authors", "Author full names", "AF", "AU"]), axis=1)

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

    out["Categorie IF"] = "empty"
    out["Categorie AIS"] = "empty"
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="Input folder with paired Scopus/WoS files")
    ap.add_argument("--output", "-o", required=True, help="Output folder for merged Excel files")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = os.listdir(args.input)
    basenames = {}
    for f in files:
        base, ext = os.path.splitext(f)
        basenames.setdefault(base, []).append(os.path.join(args.input, f))

    for base, paths in basenames.items():
        scopus_path = next((p for p in paths if p.lower().endswith(".csv")), None)
        wos_path = next((p for p in paths if p.lower().endswith((".xls", ".xlsx"))), None)

        if scopus_path and wos_path:
            scopus_df = pd.read_csv(scopus_path, dtype=str)
            wos_df = pd.read_excel(wos_path, dtype=str)
            merged_raw = combine(scopus_df, wos_df)
            formatted = format_final(merged_raw)
            out_path = os.path.join(args.output, f"{base}_merged.xlsx")
            formatted.to_excel(out_path, index=False)
            print(f"Merged {scopus_path} + {wos_path} -> {out_path}")


if __name__ == "__main__":
    main()
