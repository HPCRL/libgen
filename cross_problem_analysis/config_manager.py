"""
config_manager.py

Manages kernel configurations and problem shapes for cross-problem analysis.
Loads best configurations from CSV and provides convenient access methods.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple


@dataclass
class ProblemShape:
    """Represents a GEMM problem shape (M, N, K, L)"""
    M: int
    N: int
    K: int
    L: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.M, self.N, self.K, self.L)

    def __str__(self) -> str:
        return f"M{self.M}_N{self.N}_K{self.K}_L{self.L}"

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProblemShape):
            return False
        return self.to_tuple() == other.to_tuple()


@dataclass
class KernelConfig:
    """Represents a kernel configuration with all tunable parameters"""
    cta_m: int
    cta_n: int
    cta_k: int
    stages: int
    atom_m: int
    atom_n: int
    atom_k: int
    a_major: str
    b_major: str
    c_major: str

    def __str__(self) -> str:
        return (
            f"cta{self.cta_m}x{self.cta_n}x{self.cta_k}_"
            f"s{self.stages}_"
            f"atom{self.atom_m}x{self.atom_n}x{self.atom_k}_"
            f"{self.a_major}{self.b_major}{self.c_major}"
        )

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for easy serialization"""
        return {
            "cta_m": self.cta_m,
            "cta_n": self.cta_n,
            "cta_k": self.cta_k,
            "stages": self.stages,
            "atom_m": self.atom_m,
            "atom_n": self.atom_n,
            "atom_k": self.atom_k,
            "a_major": self.a_major,
            "b_major": self.b_major,
            "c_major": self.c_major,
        }


@dataclass
class BestConfig:
    """Associates a problem shape with its best kernel configuration"""
    problem: ProblemShape
    config: KernelConfig
    max_gflops: float
    avg_us: float


class ConfigManager:
    """
    Manages kernel configurations and problem shapes.
    Loads best configurations from CSV files.
    """

    def __init__(self, best_configs_csv: Path):
        """
        Initialize the config manager.

        Args:
            best_configs_csv: Path to CSV file with best configurations per problem
        """
        self.best_configs_csv = best_configs_csv
        self.best_configs: Dict[ProblemShape, BestConfig] = {}
        self._load_best_configs()

    def _load_best_configs(self):
        """Load best configurations from CSV file"""
        with open(self.best_configs_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                problem = ProblemShape(
                    M=int(row["M"]),
                    N=int(row["N"]),
                    K=int(row["K"]),
                    L=int(row["L"]),
                )
                config = KernelConfig(
                    cta_m=int(row["cta_m"]),
                    cta_n=int(row["cta_n"]),
                    cta_k=int(row["cta_k"]),
                    stages=int(row["stages"]),
                    atom_m=int(row["atom_m"]),
                    atom_n=int(row["atom_n"]),
                    atom_k=int(row["atom_k"]),
                    a_major=row["a_major"],
                    b_major=row["b_major"],
                    c_major=row["c_major"],
                )
                best = BestConfig(
                    problem=problem,
                    config=config,
                    max_gflops=float(row["max_gflops"]),
                    avg_us=float(row["avg_us"]),
                )
                self.best_configs[problem] = best

    def get_best_config(self, problem: ProblemShape) -> BestConfig | None:
        """Get the best configuration for a given problem shape"""
        return self.best_configs.get(problem)

    def get_all_problems(self) -> List[ProblemShape]:
        """Get all problem shapes that have best configs"""
        return list(self.best_configs.keys())

    def get_problem_subset(self, indices: List[int]) -> List[ProblemShape]:
        """Get a subset of problems by indices"""
        all_problems = self.get_all_problems()
        return [all_problems[i] for i in indices if i < len(all_problems)]

    def filter_problems(
        self,
        min_m: int | None = None,
        max_m: int | None = None,
        min_n: int | None = None,
        max_n: int | None = None,
        min_k: int | None = None,
        max_k: int | None = None,
    ) -> List[ProblemShape]:
        """Filter problems by dimension ranges"""
        filtered = []
        for problem in self.best_configs.keys():
            if min_m is not None and problem.M < min_m:
                continue
            if max_m is not None and problem.M > max_m:
                continue
            if min_n is not None and problem.N < min_n:
                continue
            if max_n is not None and problem.N > max_n:
                continue
            if min_k is not None and problem.K < min_k:
                continue
            if max_k is not None and problem.K > max_k:
                continue
            filtered.append(problem)
        return filtered

    def get_all_best_configs_for_subset(
        self, problem_subset: List[ProblemShape]
    ) -> List[BestConfig]:
        """Get best configs for a subset of problems"""
        return [self.best_configs[p] for p in problem_subset if p in self.best_configs]
