#!/usr/bin/env python3
"""Check for duplicate configs in the 5 selected problems"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager

csv_path = Path("/media/datassd/sina/libgen/cutlass-pdsl/cutlass/examples/python/CuTeDSL/ampere/collected_data/best_by_problem_v1.csv")
config_mgr = ConfigManager(csv_path)

# Rows 4, 8, 13, 18, 19 -> indices 3, 7, 12, 17, 18
problem_indices = [3, 7, 12, 17, 18]
problems = config_mgr.get_problem_subset(problem_indices)

print("Checking for duplicate configurations...\n")

configs = []
for i, problem in enumerate(problems):
    best = config_mgr.get_best_config(problem)
    config_str = str(best.config)
    configs.append((i, problem, config_str, best.config))
    print(f"[{i}] {problem}")
    print(f"    {best.config}")
    print()

# Find duplicates
print("="*80)
config_groups = {}
for i, problem, config_str, config in configs:
    if config_str not in config_groups:
        config_groups[config_str] = []
    config_groups[config_str].append((i, problem))

unique_count = len(config_groups)
total_count = len(configs)

print(f"Summary: {total_count} problems, {unique_count} unique configs")
print()

if unique_count < total_count:
    print("Duplicate configs found:")
    for config_str, problems_list in config_groups.items():
        if len(problems_list) > 1:
            print(f"\n  Config: {config_str}")
            print(f"  Used by {len(problems_list)} problems:")
            for idx, prob in problems_list:
                print(f"    [{idx}] {prob}")
    print(f"\nOptimization: Will run {len(problems) * unique_count} tests instead of {len(problems) * total_count}")
    print(f"Savings: {len(problems) * (total_count - unique_count)} redundant tests avoided")
else:
    print("✓ All configs are unique - no optimization needed")
