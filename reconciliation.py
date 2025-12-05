# reconciliation.py — Recall Boost Enhanced Version
# ---------------------------------------------------
# 提升 Recall 的主要手段：
# 1. image_num 匹配范围从 ±1 → ±2
# 2. threshold 降低到 20（默认）
# 3. defendant token overlap 权重从 12 → 18
# 4. size penalty 降低
# 5. 加强 GT place vs HTR location 匹配（新加 location 提取）
# 6. 降低 plaintiff fuzzy 权重

import json
import re
from rapidfuzz import fuzz
import numpy as np
from scipy.optimize import linear_sum_assignment


# -------------------------
# Utility
# -------------------------

def name_tokens_from_lists(name_list):
    STOPWORDS = {"of", "the", "and", "or", "de", "la", "le", "in", "at", "by", "for", "alias"}
    OCCUPATION = {"husbandman", "yeoman", "gent", "gentleman", "esq", "esquire",
                  "knight", "clerk", "merchant", "baker", "weaver", "abbot", "prior",
                  "vicar", "parson", "widow", "servant", "carpenter", "smith",
                  "laborer", "hosteler", "armorer"}

    tokens = set()
    for s in name_list:
        parts = re.split(r"[^a-z]+", s.lower())
        for p in parts:
            if len(p) < 3: 
                continue
            if p in STOPWORDS:
                continue
            if p in OCCUPATION:
                continue
            tokens.add(p)
    return tokens


def list_similarity(list1, list2):
    s1 = " ".join(list1).lower()
    s2 = " ".join(list2).lower()
    if not s1 or not s2:
        return 0
    return fuzz.token_sort_ratio(s1, s2)


def extract_image_num_from_source_dir(src):
    if not src:
        return None
    m = re.search(r"(\d+)", src)
    return int(m.group(1)) if m else None


def extract_person_names(person_list):
    names = []
    for p in person_list:
        first = p.get("firstName", "")
        last = p.get("lastName", "")
        full = f"{first} {last}".strip().lower()
        if full:
            names.append(full)
    return names


# 新增：抽取 HTR 里 defendant 的 location
def extract_locations(person_list):
    locs = []
    for p in person_list:
        loc = p.get("location", "")
        if loc:
            locs.append(loc.lower())
    return locs


def load_htr_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    out = []

    for obj in cases:
        src_dir = obj.get("source_image_directory", "")
        img = extract_image_num_from_source_dir(src_dir)

        plaintiffs = ["the crown"] if obj.get("isCrownPlaintiff") else []

        defendants = extract_person_names(obj.get("defendants", []))
        htr_locations = extract_locations(obj.get("defendants", []))

        county = (obj.get("county") or "").lower().strip()
        plea = (obj.get("plea", {}).get("primary_charge") or "").lower().strip()

        out.append({
            "image_num": img,
            "plaintiffs": plaintiffs,
            "defendants": defendants,
            "locations": htr_locations,    # 新增，提高 recall 的关键字段
            "county": county,
            "plea": plea,
        })
    return out


# -------------------------
# Similarity Score (Recall Boost)
# -------------------------

def calculate_similarity(gt, htr):
    score = 0.0

    # Plaintiff fuzzy 权重降低 (0.35 → 0.15)
    score += 0.15 * list_similarity(gt["plaintiffs"], htr["plaintiffs"])

    # Defendant fuzzy 权重保持
    score += 0.35 * list_similarity(gt["defendants"], htr["defendants"])

    # -------- 地名匹配增强 --------
    places = [p.lower() for p in gt["places"]]

    # county 匹配（原有）
    if htr["county"] and any(p in htr["county"] for p in places):
        score += 20.0

    # 新增：location 匹配（非常显著提升 recall）
    htr_loc_text = " ".join(htr.get("locations", []))
    if any(pl in htr_loc_text for pl in places):
        score += 15.0

    # -------- token overlap --------
    gt_tok = name_tokens_from_lists(gt["plaintiffs"] + gt["defendants"])
    htr_tok = name_tokens_from_lists(htr["plaintiffs"] + htr["defendants"])
    shared = gt_tok & htr_tok
    score += min(len(shared) * 18.0, 72.0)   # 权重从 12 → 18

    # -------- size mismatch penalty 降低 --------
    size_diff = abs(len(gt["defendants"]) - len(htr["defendants"]))
    score -= size_diff * 0.1                 # 从 0.3 降低到 0.1

    return max(0.0, min(100.0, score))


# -------------------------
# Reconciliation
# -------------------------

def reconcile(gt_cases, htr_cases, threshold=20.0):
    G = len(gt_cases)
    H = len(htr_cases)

    # 行数为 2G（每个 GT 两个 slot）
    cost = np.full((2 * G, H), 100.0)
    slot_info = []

    for gi, gt in enumerate(gt_cases):
        img = gt["image_num"]
        valid = []

        # image_num 宽松范围：±2（Recall 大幅提升）
        if img is not None:
            for hi, h in enumerate(htr_cases):
                if h["image_num"] is None:
                    continue
                if abs(h["image_num"] - img) <= 2:
                    valid.append(hi)

        for slot in (0, 1):
            row = 2 * gi + slot
            slot_info.append((gi, slot))

            for hi in valid:
                sim = calculate_similarity(gt, htr_cases[hi])
                cost[row][hi] = 100 - sim

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = {gi: {"matched_htr": [], "similarity": []} for gi in range(G)}

    for r, c in zip(row_ind, col_ind):
        gi, slot = slot_info[r]
        sim = 100 - cost[r][c]

        if sim >= threshold:
            matches[gi]["matched_htr"].append(c)
            matches[gi]["similarity"].append(sim)

    return matches


if __name__ == "__main__":
    print("[TEST] Recall-Boost reconciliation.py loaded successfully.")
