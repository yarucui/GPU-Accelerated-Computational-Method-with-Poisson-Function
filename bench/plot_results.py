#!/usr/bin/env python3
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def plot_global_tpb(df, out_png):
    # df columns: N, TPB, time_s, mnodes_per_s
    plt.figure(figsize=(10, 6))
    for tpb in sorted(df["TPB"].unique()):
        sub = df[df["TPB"] == tpb].sort_values("N")
        plt.plot(sub["N"], sub["time_s"], marker="o", label=f"TPB={tpb}")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Time (s)")
    plt.title("GPU Global Kernel: TPB Sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_shared_vs_global(df, out_png):
    # df columns: N, global_time_s, shared_time_s, speedup_global_over_shared
    df = df.sort_values("N")
    plt.figure(figsize=(10, 6))
    plt.plot(df["N"], df["speedup_global_over_shared"], marker="o")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Speedup (global_time / shared_time)")
    plt.title("Shared(16) vs Global(16) Speedup")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_cpu_gpu(df_cpu, df_global, df_shared, out_png):
    # Pick best TPB per N for global by max mnodes_per_s
    best_global = (
        df_global.sort_values(["N", "mnodes_per_s"], ascending=[True, False])
        .groupby("N", as_index=False)
        .first()
        .rename(columns={"mnodes_per_s": "global_best_mnodes"})
    )
    cpu = df_cpu[["N", "mnodes_per_s"]].rename(columns={"mnodes_per_s": "cpu_mnodes"})
    shared = df_shared[["N", "shared_mnodes_per_s"]].rename(columns={"shared_mnodes_per_s": "shared16_mnodes"})

    merged = cpu.merge(best_global[["N", "global_best_mnodes"]], on="N", how="inner").merge(shared, on="N", how="inner")
    merged = merged.sort_values("N")

    plt.figure(figsize=(10, 6))
    plt.plot(merged["N"], merged["cpu_mnodes"], marker="o", label="CPU NumPy")
    plt.plot(merged["N"], merged["global_best_mnodes"], marker="o", label="GPU Global (best TPB)")
    plt.plot(merged["N"], merged["shared16_mnodes"], marker="o", label="GPU Shared (TPB=16)")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("MNodes/s")
    plt.title("CPU vs GPU Throughput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Plot benchmark CSVs.")
    p.add_argument("--global_csv", type=str, default="bench/results_gpu_global.csv")
    p.add_argument("--shared_csv", type=str, default="bench/results_shared_vs_global.csv")
    p.add_argument("--cpu_csv", type=str, default="bench/results_cpu.csv")
    p.add_argument("--out_dir", type=str, default="bench/figs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_global = pd.read_csv(args.global_csv)
    df_shared = pd.read_csv(args.shared_csv)
    df_cpu = pd.read_csv(args.cpu_csv)

    plot_global_tpb(df_global, os.path.join(args.out_dir, "fig_global_tpb.png"))
    plot_shared_vs_global(df_shared, os.path.join(args.out_dir, "fig_speedup_shared_vs_global.png"))
    plot_cpu_gpu(df_cpu, df_global, df_shared, os.path.join(args.out_dir, "fig_cpu_vs_gpu.png"))

    print(f"[OK] Plots saved to: {args.out_dir}")
