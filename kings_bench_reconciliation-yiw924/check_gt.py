# check_gt.py
import json

path = "data/gt_dataset_799.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# data is dict: {"gt_case_1": {...}, ...}
first_key = next(iter(data.keys()))
first_case = data[first_key]

print("First GT key:", first_key)
print("\n=== GT case structure ===")
for k, v in first_case.items():
    print(f"{k} : {v}")
