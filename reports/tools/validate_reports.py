#!/usr/bin/env python3
"""Validate reports/ directory for common issues."""
import re, json, csv, os, sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent
ERRORS = []
WARNINGS = []

def error(msg): ERRORS.append(msg)
def warn(msg): WARNINGS.append(msg)

def check_broken_tables():
    """Check for broken Markdown tables (mismatched columns)."""
    for f in REPORTS_DIR.glob("*.md"):
        lines = f.read_text(encoding="utf-8").splitlines()
        in_table = False
        header_cols = 0
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*\|.*\|', line):
                cols = len([c for c in line.split('|') if c.strip() != '']) if not line.strip().startswith('|') else len(line.split('|')) - 2
                if not in_table:
                    in_table = True
                    header_cols = cols
                elif re.match(r'^\s*\|[\s\-:|]+\|', line):
                    continue  # separator row
                else:
                    if cols != header_cols and header_cols > 0:
                        warn(f"{f.name}:{i} table column mismatch (expected {header_cols}, got {cols})")
            else:
                in_table = False
                header_cols = 0

def check_autolink_pollution():
    """Check for broken auto-links like [Text](url)ion."""
    pattern = re.compile(r'\]\([^)]+\)[a-zA-Z]{2,}')
    for f in REPORTS_DIR.glob("*.md"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                error(f"{f.name}:{i} auto-link pollution: ...{m.group()[:40]}")

def check_empty_links():
    """Check for empty links []() or [text]()."""
    pattern = re.compile(r'\[[^\]]*\]\(\s*\)')
    for f in REPORTS_DIR.glob("*.md"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                error(f"{f.name}:{i} empty link: {m.group()}")

def check_duplicate_headings():
    """Check for duplicate headings within same file."""
    for f in REPORTS_DIR.glob("*.md"):
        headings = []
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r'^(#{1,4})\s+(.+)', line)
            if m:
                h = m.group(2).strip()
                if h in headings:
                    warn(f"{f.name}:{i} duplicate heading: {h}")
                headings.append(h)

def check_master_index():
    """Check that master report file index line counts are accurate."""
    master = REPORTS_DIR / "00_MASTER_REPORT.md"
    if not master.exists():
        error("00_MASTER_REPORT.md not found")
        return
    content = master.read_text(encoding="utf-8")
    # Find lines referencing report files with line counts
    for f in REPORTS_DIR.glob("*.md"):
        if f.name == "00_MASTER_REPORT.md":
            continue
        actual = len(f.read_text(encoding="utf-8").splitlines())
        # Check if master mentions this file
        if f.name not in content:
            warn(f"00_MASTER_REPORT.md missing reference to {f.name}")

def check_catalog_consistency():
    """Check CSV and JSON catalog entry counts match."""
    csv_file = REPORTS_DIR / "02_paper_catalog.csv"
    json_file = REPORTS_DIR / "02_paper_catalog.json"
    if csv_file.exists():
        with open(csv_file, encoding="utf-8") as f:
            csv_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
    if json_file.exists():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        json_count = data.get("metadata", {}).get("total_papers", 0)
        if csv_rows != json_count:
            warn(f"Catalog mismatch: CSV has {csv_rows} rows, JSON metadata says {json_count}")
    # Check enriched versions
    ecsv = REPORTS_DIR / "02_paper_catalog_enriched.csv"
    ejson = REPORTS_DIR / "02_paper_catalog_enriched.json"
    if ecsv.exists() and csv_file.exists():
        with open(ecsv, encoding="utf-8") as f:
            ecsv_rows = sum(1 for _ in csv.reader(f)) - 1
        if ecsv_rows != csv_rows:
            warn(f"Enriched CSV has {ecsv_rows} rows vs original {csv_rows}")

def main():
    print("=" * 60)
    print("Reports Validation")
    print("=" * 60)
    check_broken_tables()
    check_autolink_pollution()
    check_empty_links()
    check_duplicate_headings()
    check_master_index()
    check_catalog_consistency()
    print(f"\nErrors: {len(ERRORS)}")
    for e in ERRORS:
        print(f"  ❌ {e}")
    print(f"\nWarnings: {len(WARNINGS)}")
    for w in WARNINGS:
        print(f"  ⚠️  {w}")
    print(f"\n{'PASS' if not ERRORS else 'FAIL'} — {len(ERRORS)} errors, {len(WARNINGS)} warnings")
    return 1 if ERRORS else 0

if __name__ == "__main__":
    sys.exit(main())
