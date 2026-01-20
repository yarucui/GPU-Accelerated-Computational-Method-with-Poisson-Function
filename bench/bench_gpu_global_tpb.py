#!/usr/bin/env python3
import os
import math
import csv
import argparse
import time

import numpy as np

try:
    import cupy as cp
except Exception as e:
    raise SystemExit(f"[ERROR] Cannot import cupy: {e}")


CUDA_SRC_GLOBAL = r'''
extern "C" __global__
void poisson_global_kernel(const float* u_in, float* u_out, const float* b, int N, int M) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * M + x;

    if (x > 0 && x < M - 1 && y > 0 && y < N - 1) {
        float val = u_in[(y - 1) * M + x] + u_in[(y + 1) * M + x]
                  + u_in[y * M + (x - 1)] + u_in[y * M + (x + 1)];
        u_out[idx] = 0.25f * (val - b[idx]);
    }
}
'''


def gpu_name() -> str:
    props = cp.cuda.runtime.getDeviceProperties(0)
    return props["name"].decode("utf-8")


def time_kernel(kernel, grid_dim, block_dim, args, iters: int, warmup: int = 10) -> float:
    # warmup
    for _ in range(warmup):
        kernel(grid_dim, block_dim, args)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for i in range(iters):
        kernel(grid_dim, block_dim, args)
    end.record()
    end.synchronize()

    # milliseconds -> seconds
    ms = cp.cuda.get_elapsed_time(start, end)
    return ms / 1000.0


def run(n_list, tpb_list, iters, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    kernel = cp.RawKernel(CUDA_SRC_GLOBAL, "poisson_global_kernel")

    print(f"GPU: {gpu_name()}")
    print("\n=== GPU Global Kernel: TPB Sweep ===")
    print(f"{'N':<8} | {'TPB':<4} | {'Time(s)':<12} | {'MNodes/s':<12}")
    print("-" * 55)

    rows = []
    for n in n_list:
        for tpb in tpb_list:
            u = cp.zeros((n, n), dtype=cp.float32)
            b = cp.random.random((n, n), dtype=cp.float32)
            u_out = cp.zeros_like(u)

            block_dim = (tpb, tpb)
            grid_dim = (int(math.ceil(n / tpb)), int(math.ceil(n / tpb)))

            # ping-pong buffers to avoid extra allocations
            # NOTE: for global kernel, args must be passed each call; RawKernel packs pointers.
            # We'll time an "alternate" loop by calling kernel with alternating u/u_out in Python
            # to match your original benchmark logic.
            warmup = 10
            for _ in range(warmup):
                kernel(grid_dim, block_dim, (u, u_out, b, n, n))
                kernel(grid_dim, block_dim, (u_out, u, b, n, n))
            cp.cuda.Stream.null.synchronize()

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for i in range(iters):
                if i % 2 == 0:
                    kernel(grid_dim, block_dim, (u, u_out, b, n, n))
                else:
                    kernel(grid_dim, block_dim, (u_out, u, b, n, n))
            end.record()
            end.synchronize()

            total_s = cp.cuda.get_elapsed_time(start, end) / 1000.0
            mnodes = (n * n * iters) / total_s / 1e6

            print(f"{n:<8} | {tpb:<4} | {total_s:<12.6f} | {mnodes:<12.2f}")

            rows.append({
                "N": n,
                "TPB": tpb,
                "iters": iters,
                "time_s": total_s,
                "mnodes_per_s": mnodes,
            })

            del u, b, u_out
            cp.get_default_memory_pool().free_all_blocks()

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[OK] Wrote: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="GPU Global Kernel TPB sweep benchmark (Poisson stencil).")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    p.add_argument("--tpb", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", type=str, default="bench/results_gpu_global.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sizes, args.tpb, args.iters, args.out)
