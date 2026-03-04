from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src.config_io import load_config, save_config
from src.gridworld import GridWorld
from src.maze_generator import generate_maze_config
from src.policy_iteration import policy_iteration
from src.reporting import compile_report, write_report
from src.utils import (
    build_state_utility_series,
    compare_policies,
    dump_metrics_json,
    extract_policy,
    policy_to_grid,
    utilities_to_matrix,
    write_matrix_csv,
    write_policy_txt,
)
from src.value_iteration import value_iteration
from src.visualization import (
    plot_series_png,
    save_board_png,
    save_policy_png,
    save_utility_heatmap_png,
)


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"


def build_env(config_path: Path) -> GridWorld:
    return GridWorld.from_config(load_config(config_path))


def run_part1(config_path: Path) -> dict:
    env = build_env(config_path)
    output_dir = RESULTS_DIR / "part1"
    output_dir.mkdir(parents=True, exist_ok=True)

    vi_start = time.perf_counter()
    vi_utilities, vi_history = value_iteration(env)
    vi_runtime_ms = (time.perf_counter() - vi_start) * 1000.0
    vi_policy = extract_policy(env, vi_utilities)

    pi_start = time.perf_counter()
    pi_policy, pi_utilities, pi_history = policy_iteration(env)
    pi_runtime_ms = (time.perf_counter() - pi_start) * 1000.0

    vi_matrix = utilities_to_matrix(env, vi_utilities)
    pi_matrix = utilities_to_matrix(env, pi_utilities)

    write_matrix_csv(output_dir / "value_iteration_utilities.csv", vi_matrix)
    write_matrix_csv(output_dir / "policy_iteration_utilities.csv", pi_matrix)
    write_policy_txt(output_dir / "value_iteration_policy.txt", env, vi_policy)
    write_policy_txt(output_dir / "policy_iteration_policy.txt", env, pi_policy)

    save_board_png(env, output_dir / "maze.png", title="Base Maze")
    save_policy_png(
        env,
        vi_policy,
        vi_utilities,
        output_dir / "value_iteration_policy.png",
        title="Value Iteration Policy",
    )
    save_policy_png(
        env,
        pi_policy,
        pi_utilities,
        output_dir / "policy_iteration_policy.png",
        title="Policy Iteration Policy",
    )
    save_utility_heatmap_png(
        env,
        vi_utilities,
        output_dir / "value_iteration_utility.png",
        title="Value Iteration Utilities",
    )
    save_utility_heatmap_png(
        env,
        pi_utilities,
        output_dir / "policy_iteration_utility.png",
        title="Policy Iteration Utilities",
    )
    plot_series_png(
        build_state_utility_series(env, vi_history),
        output_dir / "value_iteration_utility_estimates.png",
        title="Value Iteration Utility Estimates by State",
        x_label="Iteration",
        y_label="Utility Estimate",
    )
    plot_series_png(
        build_state_utility_series(env, pi_history),
        output_dir / "policy_iteration_utility_estimates.png",
        title="Policy Iteration Utility Estimates by State",
        x_label="Iteration",
        y_label="Utility Estimate",
    )

    max_utility_diff = max(
        abs(vi_utilities[state] - pi_utilities[state]) for state in env.get_states()
    )
    policy_mismatches = compare_policies(env, vi_policy, pi_policy)

    part1_summary = {
        "config": str(config_path.relative_to(ROOT)),
        "value_iteration": {
            "iterations": len(vi_history),
            "runtime_ms": round(vi_runtime_ms, 3),
            "mean_utility": round(vi_history[-1]["mean_utility"], 6),
            "delta": vi_history[-1]["delta"],
        },
        "policy_iteration": {
            "iterations": len(pi_history),
            "runtime_ms": round(pi_runtime_ms, 3),
            "mean_utility": round(pi_history[-1]["mean_utility"], 6),
            "delta": pi_history[-1]["delta"],
        },
        "comparison": {
            "max_utility_diff": max_utility_diff,
            "policy_mismatches": policy_mismatches,
        },
    }
    dump_metrics_json(output_dir / "summary.json", part1_summary)
    return part1_summary


def run_part2(seed: int, sizes: list[int]) -> dict:
    output_dir = RESULTS_DIR / "part2"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    iteration_series_vi = []
    iteration_series_pi = []
    runtime_series_vi = []
    runtime_series_pi = []
    mean_utility_series_vi = []
    mean_utility_series_pi = []

    for size in sizes:
        config = generate_maze_config(size=size, seed=seed + size)
        config_path = CONFIG_DIR / f"maze_{size}.yaml"
        save_config(config_path, config)

        env = GridWorld.from_config(config)
        maze_dir = output_dir / f"maze_{size}"
        maze_dir.mkdir(parents=True, exist_ok=True)

        save_board_png(env, maze_dir / "maze.png", title=f"Generated Maze {size}x{size}")

        vi_start = time.perf_counter()
        vi_utilities, vi_history = value_iteration(env)
        vi_runtime_ms = (time.perf_counter() - vi_start) * 1000.0
        vi_policy = extract_policy(env, vi_utilities)

        pi_start = time.perf_counter()
        pi_policy, pi_utilities, pi_history = policy_iteration(env)
        pi_runtime_ms = (time.perf_counter() - pi_start) * 1000.0

        save_policy_png(
            env,
            vi_policy,
            vi_utilities,
            maze_dir / "value_iteration_policy.png",
            title=f"Value Iteration Policy {size}x{size}",
        )
        save_policy_png(
            env,
            pi_policy,
            pi_utilities,
            maze_dir / "policy_iteration_policy.png",
            title=f"Policy Iteration Policy {size}x{size}",
        )
        save_utility_heatmap_png(
            env,
            vi_utilities,
            maze_dir / "value_iteration_utility.png",
            title=f"Value Iteration Utilities {size}x{size}",
        )
        save_utility_heatmap_png(
            env,
            pi_utilities,
            maze_dir / "policy_iteration_utility.png",
            title=f"Policy Iteration Utilities {size}x{size}",
        )

        row = {
            "maze_size": size,
            "value_iteration_iterations": len(vi_history),
            "value_iteration_runtime_ms": round(vi_runtime_ms, 3),
            "value_iteration_mean_utility": round(vi_history[-1]["mean_utility"], 6),
            "policy_iteration_iterations": len(pi_history),
            "policy_iteration_runtime_ms": round(pi_runtime_ms, 3),
            "policy_iteration_mean_utility": round(pi_history[-1]["mean_utility"], 6),
            "policy_mismatches": compare_policies(env, vi_policy, pi_policy),
        }
        summary_rows.append(row)

        iteration_series_vi.append((size, len(vi_history)))
        iteration_series_pi.append((size, len(pi_history)))
        runtime_series_vi.append((size, round(vi_runtime_ms, 3)))
        runtime_series_pi.append((size, round(pi_runtime_ms, 3)))
        mean_utility_series_vi.append((size, round(vi_history[-1]["mean_utility"], 6)))
        mean_utility_series_pi.append((size, round(pi_history[-1]["mean_utility"], 6)))

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_series_png(
        {
            "Value Iteration": iteration_series_vi,
            "Policy Iteration": iteration_series_pi,
        },
        output_dir / "iterations_vs_size.png",
        title="Maze Size vs Iterations",
        x_label="Maze Size",
        y_label="Iterations",
    )
    plot_series_png(
        {
            "Value Iteration": runtime_series_vi,
            "Policy Iteration": runtime_series_pi,
        },
        output_dir / "runtime_vs_size.png",
        title="Maze Size vs Runtime",
        x_label="Maze Size",
        y_label="Runtime (ms)",
    )
    plot_series_png(
        {
            "Value Iteration": mean_utility_series_vi,
            "Policy Iteration": mean_utility_series_pi,
        },
        output_dir / "mean_utility_vs_size.png",
        title="Maze Size vs Mean Utility",
        x_label="Maze Size",
        y_label="Mean Utility",
    )

    summary = {
        "seed": seed,
        "sizes": sizes,
        "rows": summary_rows,
    }
    dump_metrics_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 1 MDP Maze Solver")
    parser.add_argument(
        "--part",
        choices=["1", "2", "all"],
        default="all",
        help="Which assignment part to run.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "base.yaml"),
        help="Base config path for Part 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic Part 2 maze generation.",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[7, 9, 11, 13, 15, 17],
        help="Maze sizes used for Part 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part1_summary = None
    part2_summary = None

    if args.part in {"1", "all"}:
        part1_summary = run_part1(Path(args.config))

    if args.part in {"2", "all"}:
        part2_summary = run_part2(args.seed, args.sizes)

    report_path = write_report(ROOT, part1_summary=part1_summary, part2_summary=part2_summary)
    pdf_path = None
    try:
        pdf_path = compile_report(report_path)
    except Exception as exc:
        print(json.dumps({"warning": f"LaTeX compilation failed: {exc}"}, indent=2))

    summary = {
        "part1": part1_summary,
        "part2": part2_summary,
        "report_tex": str(report_path.relative_to(ROOT)),
        "report_pdf": str(pdf_path.relative_to(ROOT)) if pdf_path else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
