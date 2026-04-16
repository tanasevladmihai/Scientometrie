#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import re
import glob
import logging
from functionalities.common import (
    to_text, normalize_title, smart_read_excel, infer_year_from_filename
)

logger = logging.getLogger(__name__)

# ---------- Utilities ----------

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
    v = re.sub(r"[^a-z0-9]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

# ---------- helpers for picking "best" scores ----------

def _best_quartile(vals):
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
    vals = [to_text(v) for v in vals if to_text(v)]
    if not vals:
        return ""
    any_quart = any(str(v).strip().upper().startswith("Q") for v in vals)
    return _best_quartile(vals) if any_quart else _best_numeric(vals)

# ---------- year-fallback lookup ----------

def _lookup_journal_best(jidx, norm, year):
    sub = jidx[jidx["norm_title"] == norm]
    if sub.empty or pd.isna(year):
        return "", ""
    y = int(year)

    exact = sub[sub["year"] == y]
    if not exact.empty:
        chosen = exact.iloc[0]
        return to_text(chosen.get("if", "")), to_text(chosen.get("ais", ""))

    prior = sub[sub["year"] < y]
    if prior.empty:
        return "", ""
    chosen = prior.sort_values("year", ascending=False).iloc[0]
    return to_text(chosen.get("if", "")), to_text(chosen.get("ais", ""))


def _lookup_core_best(cidx, norm, year):
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


# ---------- build indexes ----------

def build_journal_index(journal_dir: str):
    frames = []
    for path in glob.glob(os.path.join(journal_dir, "normalized_*.xlsx")):
        year_str = infer_year_from_filename(path)
        if year_str == "N/A": continue
        year = int(year_str)
        
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

    def agg_best(g):
        return pd.Series({
            "if": _best_if_or_quartile(g["if"].tolist()),
            "ais": _best_quartile(g["ais"].tolist())
        })

    if not jidx.empty:
        jidx = jidx.groupby(["norm_title", "year"], as_index=False).apply(
            agg_best, include_groups=False
        )
        jidx["year"] = jidx["year"].astype(int)
    return jidx


def build_core_index(core_dir: str):
    frames = []
    for path in glob.glob(os.path.join(core_dir, "core_normalized_*.csv")):
        year_str = infer_year_from_filename(path)
        if year_str == "N/A": continue
        year = int(year_str)
        
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


def lookup_scores(title_raw, year, jidx, cidx):
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
    df = df.drop_duplicates(
        subset=["Titlu", "Autori", "FORUM (Revista, Conferința)", "An"],
        keep="first"
    ).reset_index(drop=True)
    return df


def merge_possible(df_row, candidates, clean_authors=False):
    vals = []
    seen = set()
    for c in candidates:
        if c in df_row.index and "id" not in c.lower():
            v = to_text(df_row[c])
            if v:
                if clean_authors:
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

    if vol: parts.append(f"Vol {vol}")
    if iss: parts.append(f"No {iss}")
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
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv", ".txt"}:
        try:
            df = pd.read_csv(path, dtype=str)
            if df.shape[1] > 1: return df
        except Exception: pass
        for sep in [",", "\t", ";", "|"]:
            for enc in ["utf-8", "utf-8-sig", "latin1", "utf-16", "utf-16le"]:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
                    if df.shape[1] > 1: return df
                except Exception: continue
        raise RuntimeError(f"Unable to parse text file: {path}")

    return smart_read_excel(path, dtype=str)

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


def main(input_dir, output_dir, file_list=None, journal_dir="out/journal", core_dir="out/core"):
    os.makedirs(output_dir, exist_ok=True)
    files_to_process = file_list if file_list else os.listdir(input_dir)
    basenames = {}
    for f in files_to_process:
        base, ext = os.path.splitext(f)
        basenames.setdefault(base, []).append(os.path.join(input_dir, f))

    for base, paths in basenames.items():
        scopus_path = next((p for p in paths if p.lower().endswith(".csv")), None)
        wos_path = next((p for p in paths if p.lower().endswith((".xls", ".xlsx"))), None)

        if scopus_path and wos_path:
            scopus_df = pd.read_csv(scopus_path, dtype=str)
            wos_df = smart_read_excel(wos_path, dtype=str)
            raw = combine(scopus_df, wos_df)
            suffix = "_merged"
        elif scopus_path or wos_path:
            raw = read_any_table(scopus_path or wos_path)
            suffix = "_single"
        else: continue

        formatted = format_final(raw)
        formatted = attach_scores(formatted, journal_dir=journal_dir, core_dir=core_dir)
        out_path = os.path.join(output_dir, f"{base}{suffix}.xlsx")
        formatted.to_excel(out_path, index=False)
        logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="exports")
    ap.add_argument("--output", "-o", default="out")
    ap.add_argument("--journal_dir", default="out/journal")
    ap.add_argument("--core_dir", default="out/core")
    ap.add_argument('files', nargs='*')
    args = ap.parse_args()
    main(args.input, args.output, file_list=args.files if args.files else None, 
         journal_dir=args.journal_dir, core_dir=args.core_dir)
