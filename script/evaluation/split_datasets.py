#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch split datasets in 'bench_out' into 'train' and 'test' directories (80:20 split).
Also extracts and saves unique official labels to 'label.txt'.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent
BENCH_OUT_DIR = BASE_DIR / "bench_out"
TRAIN_ROOT = BASE_DIR / "train"
TEST_ROOT = BASE_DIR / "test"
LABEL_FILE = BASE_DIR / "label.txt"

RANDOM_SEED = 42
TEST_RATIO = 0.2

def get_label_official(d):
    """
    Extract 'official' label from data item.
    Prioritizes root 'label' key (bench_out format), falls back to nested 'json' -> 'label'.
    """
    # 1. Try root level 'label' (Standard bench_out format)
    lbl = d.get("label")
    if lbl and isinstance(lbl, dict) and "official" in lbl:
        return lbl["official"]
    
    # 2. Try inside 'json' (Raw data format often seen in list_split_json.py)
    j = d.get("json")
    if isinstance(j, dict):
        lbl = j.get("label")
        if lbl and isinstance(lbl, dict) and "official" in lbl:
            return lbl["official"]
            
    # 3. Handle list wrapper inside 'json' if necessary
    if isinstance(j, list) and len(j) > 0:
        lbl = j[0].get("label")
        if lbl and isinstance(lbl, dict) and "official" in lbl:
            return lbl["official"]

    return "unknown"

def write_json(path, obj):
    """Helper to write JSON file"""
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def split_dataset(dataset_dir):
    """
    Process a single dataset directory.
    Returns (dataset_name, unique_labels_list) or None if failed.
    """
    dataset_name = dataset_dir.name
    json_file = dataset_dir / f"{dataset_name}.json"
    
    if not json_file.exists():
        print(f"[SKIP] JSON file not found for {dataset_name}: {json_file}")
        return None

    print(f"\nProcessing {dataset_name}...")
    
    try:
        with json_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {json_file}: {e}")
        return None

    # Handle if data is wrapped in a dict under 'data' key
    if isinstance(data, dict):
        data = data.get('data', [])
    
    if not isinstance(data, list):
        print(f"[ERROR] Data in {json_file} is not a list.")
        return None

    if not data:
        print(f"[WARN] No data found in {json_file}")
        return None

    # Group by label
    buckets = defaultdict(list)
    unique_labels = set()
    
    for d in data:
        official_label = get_label_official(d)
        buckets[official_label].append(d)
        unique_labels.add(official_label)

    # Statistics
    print(f"  Total samples: {len(data)}")
    print(f"  Unique classes: {len(buckets)}")
    # print(f"  Class distribution: {dict(Counter(officials))}")

    train_samples = []
    test_samples = []

    # Stratified Split
    random.seed(RANDOM_SEED)
    
    for label, bucket in buckets.items():
        random.shuffle(bucket)
        total = len(bucket)
        n_test = max(1, int(total * TEST_RATIO))
        
        # Edge case: if total is 1, n_test becomes 1, leaving train empty. 
        # This matches list_split_json.py logic. 
        
        test_subset = bucket[:n_test]
        train_subset = bucket[n_test:]
        
        test_samples.extend(test_subset)
        train_samples.extend(train_subset)

    # Prepare output directories
    dataset_train_dir = TRAIN_ROOT / dataset_name
    dataset_test_dir = TEST_ROOT / dataset_name
    
    dataset_train_dir.mkdir(parents=True, exist_ok=True)
    dataset_test_dir.mkdir(parents=True, exist_ok=True)

    # Save
    train_file = dataset_train_dir / f"{dataset_name}.json"
    test_file = dataset_test_dir / f"{dataset_name}.json"
    
    write_json(train_file, train_samples)
    write_json(test_file, test_samples)
    
    print(f"  -> Saved Train: {len(train_samples)} samples to {train_file}")
    print(f"  -> Saved Test:  {len(test_samples)} samples to {test_file}")

    return dataset_name, sorted(list(unique_labels))

def main():
    if not BENCH_OUT_DIR.exists():
        sys.exit(f"Error: bench_out directory not found at {BENCH_OUT_DIR}")

    print(f"Source Directory: {BENCH_OUT_DIR}")
    print(f"Train Output:     {TRAIN_ROOT}")
    print(f"Test Output:      {TEST_ROOT}")
    print(f"Label File:       {LABEL_FILE}")
    print("-" * 50)

    # Iterate over all directories in bench_out
    dataset_dirs = [d for d in BENCH_OUT_DIR.iterdir() if d.is_dir()]
    
    if not dataset_dirs:
        print("No dataset directories found in bench_out.")
        return

    all_dataset_labels = {}

    for dataset_dir in sorted(dataset_dirs):
        result = split_dataset(dataset_dir)
        if result:
            name, labels = result
            all_dataset_labels[name] = labels
    
    # Write label.txt
    print(f"\nWriting labels to {LABEL_FILE}...")
    try:
        with LABEL_FILE.open('w', encoding='utf-8') as f:
            # Writing in valid Python dictionary format line by line, or just lines
            # Requirement: 'NF-ToN-IoT-v2': ['Benign', ...],
            for name in sorted(all_dataset_labels.keys()):
                labels = all_dataset_labels[name]
                f.write(f"'{name}': {labels},\n")
        print("Labels saved successfully.")
    except Exception as e:
        print(f"Failed to write label file: {e}")
    
    print("\n" + "=" * 50)
    print("All done.")

if __name__ == "__main__":
    main()