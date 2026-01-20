#!/usr/bin/env python3
import os
import csv
import argparse
import time

import numpy as np


def cpu_jacobi_step(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    # vectorized slicing (no Python loop over cells)
    u_out = u.copy()
    u_out[1:-1, 1:-1] = 0.25 * (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - b[1:-1, 1:-1]
    )
    return u_out


def run(n_list, iters, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print("\n=== CPU NumPy Baseline ===")
    print(f"{'N':<8} | {'Time(s)':<12} | {'MNodes/s':<12}")
    print("-" * 40)

    rows = []
    for n in n_list:
        u = np.zeros((n, n), dtype=np.float32)
        b = np.random.rand(n, n).astype(np.float32)

        t0 = time.time()
        for _ in range(iters):
            u = cpu_jacobi_step(u, b)
        t1 = time.time()

        total_s = t1 - t0
        mnodes = (n * n * iters) / total_s / 1e6

        print(f"{n:<8} | {total_s:<12.6f} | {mnodes:<12.2f}")

        rows.append({
            "N": n,
            "iters": iters,
            "time_s": total_s,
            "mnodes_per_s": mnodes,
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[OK] Wrote: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="CPU NumPy baseline benchmark (vectorized Poisson stencil).")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    # CPU iters should be lower for big N
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", type=str, default="bench/results_cpu.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sizes, args.iters, args.out)
