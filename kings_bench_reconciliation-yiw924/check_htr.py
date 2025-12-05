# check_htr.py
import json

path = "data/htr_dataset_799.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data["cases"]

print("Total cases:", len(cases))
print("Type of first case:", type(cases[0]))

print("\n=== First case keys ===")
print(cases[0].keys())

print("\n=== First case full content ===")
print(cases[0])
