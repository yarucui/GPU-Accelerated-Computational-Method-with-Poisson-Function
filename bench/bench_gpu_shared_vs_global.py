#!/usr/bin/env python3
import os
import math
import csv
import argparse

import numpy as np

try:
    import cupy as cp
except Exception as e:
    raise SystemExit(f"[ERROR] Cannot import cupy: {e}")


CUDA_SRC_SHARED_16 = r'''
extern "C" __global__
void poisson_shared_kernel(const float* u_in, float* u_out, const float* b, int N, int M) {
    // Static Shared Memory: 16x16 block + 2 halo = 18x18
    __shared__ float s_u[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int idx = y * M + x;

    int sx = tx + 1;
    int sy = ty + 1;

    // Load center
    s_u[sy][sx] = 0.0f;
    if (y < N && x < M) {
        s_u[sy][sx] = u_in[idx];
    }

    // Halo
    if (ty == 0 && y > 0)        s_u[0][sx]  = u_in[(y - 1) * M + x];
    if (ty == 15 && y < N - 1)   s_u[17][sx] = u_in[(y + 1) * M + x];
    if (tx == 0 && x > 0)        s_u[sy][0]  = u_in[y * M + (x - 1)];
    if (tx == 15 && x < M - 1)   s_u[sy][17] = u_in[y * M + (x + 1)];

    __syncthreads();

    if (y > 0 && y < N - 1 && x > 0 && x < M - 1) {
        float val = s_u[sy - 1][sx] + s_u[sy + 1][sx] + s_u[sy][sx - 1] + s_u[sy][sx + 1];
        u_out[idx] = 0.25f * (val - b[idx]);
    }
}
'''

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


def run(n_list, iters, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    shared_kernel = cp.RawKernel(CUDA_SRC_SHARED_16, "poisson_shared_kernel")
    global_kernel = cp.RawKernel(CUDA_SRC_GLOBAL, "poisson_global_kernel")

    TPB = 16
    print(f"GPU: {gpu_name()}")
    print("\n=== GPU Shared(16) vs Global(16) ===")
    print(f"{'N':<8} | {'Global(s)':<12} | {'Shared(s)':<12} | {'Speedup':<8}")
    print("-" * 60)

    rows = []
    for n in n_list:
        u = cp.zeros((n, n), dtype=cp.float32)
        b = cp.random.random((n, n), dtype=cp.float32)
        u_out = cp.zeros_like(u)

        block_dim = (TPB, TPB)
        grid_dim = (int(math.ceil(n / TPB)), int(math.ceil(n / TPB)))

        # warmup both
        for _ in range(10):
            global_kernel(grid_dim, block_dim, (u, u_out, b, n, n))
            shared_kernel(grid_dim, block_dim, (u, u_out, b, n, n))
        cp.cuda.Stream.null.synchronize()

        # time global
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for i in range(iters):
            if i % 2 == 0:
                global_kernel(grid_dim, block_dim, (u, u_out, b, n, n))
            else:
                global_kernel(grid_dim, block_dim, (u_out, u, b, n, n))
        end.record()
        end.synchronize()
        t_global = cp.cuda.get_elapsed_time(start, end) / 1000.0

        # time shared
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for i in range(iters):
            if i % 2 == 0:
                shared_kernel(grid_dim, block_dim, (u, u_out, b, n, n))
            else:
                shared_kernel(grid_dim, block_dim, (u_out, u, b, n, n))
        end.record()
        end.synchronize()
        t_shared = cp.cuda.get_elapsed_time(start, end) / 1000.0

        mnodes_global = (n * n * iters) / t_global / 1e6
        mnodes_shared = (n * n * iters) / t_shared / 1e6
        speedup = t_global / t_shared if t_shared > 0 else float("nan")

        print(f"{n:<8} | {t_global:<12.6f} | {t_shared:<12.6f} | {speedup:<8.2f}")

        rows.append({
            "N": n,
            "TPB": TPB,
            "iters": iters,
            "global_time_s": t_global,
            "shared_time_s": t_shared,
            "global_mnodes_per_s": mnodes_global,
            "shared_mnodes_per_s": mnodes_shared,
            "speedup_global_over_shared": speedup,
        })

        del u, b, u_out
        cp.get_default_memory_pool().free_all_blocks()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[OK] Wrote: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="GPU shared(16) vs global(16) benchmark.")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", type=str, default="bench/results_shared_vs_global.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sizes, args.iters, args.out)
