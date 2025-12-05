# scraper.py
# Parse cached HTML and provide load_gt_cases() for analysis

from bs4 import BeautifulSoup
import json
import re

CACHE_PATH = "data/kb799_cached.html"

def parse_image_num(txt):
    """Extract digits from image text."""
    m = re.search(r"(\d+)", txt)
    return int(m.group(1)) if m else None

def split_names(raw: str):
    """Split names by comma or semicolon."""
    if not raw:
        return []
    parts = re.split(r"[;,]", raw)
    return [s.strip().lower() for s in parts if s.strip()]

def split_places(raw: str):
    """Split places by comma or semicolon."""
    if not raw:
        return []
    parts = re.split(r"[;,]", raw)
    return [p.strip() for p in parts if p.strip()]

def scrape_cached_html():
    """Scrape HTML into structured GT dataset."""
    print("Loading cached HTML:", CACHE_PATH)

    with open(CACHE_PATH, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    rows = soup.find_all("tr")
    print(f"Found {len(rows)} <tr> rows.")

    gt_cases = []
    case_id = 1

    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) != 5:  # Only valid data rows
            continue

        c0 = cells[0].get_text(" ", strip=True)  # image
        c1 = cells[1].get_text(" ", strip=True)  # county / place
        c2 = cells[2].get_text(" ", strip=True)  # plaintiff
        c3 = cells[3].get_text(" ", strip=True)  # defendant
        c4 = cells[4].get_text(" ", strip=True)  # plea

        gt_cases.append({
            "id": f"gt_case_{case_id}",
            "image_text": c0,
            "image_num": parse_image_num(c0),
            "plaintiffs": split_names(c2),
            "defendants": split_names(c3),
            "places": split_places(c1),
            "plea": c4.strip().lower(),
        })
        case_id += 1

    print(f"✓ Parsed {len(gt_cases)} GT cases.")
    return gt_cases

def save_json(gt_cases, path="data/gt_dataset_scraped.json"):
    """Save GT cases to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt_cases, f, indent=2)
    print("Saved:", path)

def load_gt_cases(path="data/gt_dataset_scraped.json"):
    """Load scraped GT dataset with cleaned fields."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_cases = []
    for case in data:
        gt_cases.append({
            "id": case.get("id"),
            "image_num": case.get("image_num"),
            "image_text": case.get("image_text"),
            "plaintiffs": [p.lower().strip() for p in case.get("plaintiffs", [])],
            "defendants": [d.lower().strip() for d in case.get("defendants", [])],
            "places": [p.lower().strip() for p in case.get("places", [])],
            "plea": (case.get("plea") or "").lower().strip(),
        })
    return gt_cases

def main():
    """Main function for manual scraping."""
    gt_cases = scrape_cached_html()
    save_json(gt_cases)

if __name__ == "__main__":
    main()