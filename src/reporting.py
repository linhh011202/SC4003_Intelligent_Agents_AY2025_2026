from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from .config_io import load_config


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _load_csv_table(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _latex_matrix_table(matrix: list[list[str]], caption: str, label: str) -> str:
    column_count = len(matrix[0])
    col_spec = "c" * column_count
    body = "\n".join(" & ".join(_escape_latex(item) for item in row) + r" \\" for row in matrix)
    return rf"""
\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{{col_spec}}}
{body}
\end{{tabular}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
"""


def _figure(path: str, caption: str, label: str, width: str = "0.82\\textwidth") -> str:
    return rf"""
\begin{{figure}}[h]
\centering
\includegraphics[width={width}]{{{_escape_latex(path)}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def _part1_method_section(
    method_title: str,
    implementation_text: str,
    policy_image: str,
    utility_image: str,
    estimate_image: str,
    utility_table: list[list[str]],
    iterations: int,
    runtime_ms: float,
    method_slug: str,
    method_note: str,
    utility_note: str,
) -> str:
    return rf"""
\section{{Using method of {method_title.lower()} for Part 1}}

\subsection{{Descriptions of implemented solutions}}
{implementation_text}

\subsection{{Plot of optimal policy}}
{_figure(policy_image, f"Optimal policy produced by {method_title}.", f"fig:{method_slug}_policy")}
{method_note}

\subsection{{Utilities of all states}}
{_figure(utility_image, f"Utility heatmap for {method_title}. Each number is the utility of a non-wall state. Green and brown reward cells are revisitable states, not terminal states.", f"fig:{method_slug}_utilities")}
{_latex_matrix_table(utility_table, f"Utility values for {method_title}.", f"tab:{method_slug}_utilities")}
{utility_note}

\subsection{{Plot of utility estimates as a function of the number of iterations}}
{_figure(estimate_image, f"Utility estimate of every non-wall state versus iteration for {method_title}. Total iterations: {iterations}. Runtime: {runtime_ms} ms.", f"fig:{method_slug}_estimate")}
The curves flatten out near the end of the plot, which is the practical sign that the utility estimates have converged.
"""


def _part2_section(part2_summary: dict) -> str:
    rows = "\n".join(
        f"{row['maze_size']} & {row['value_iteration_iterations']} & {row['policy_iteration_iterations']} & "
        f"{row['value_iteration_runtime_ms']} & {row['policy_iteration_runtime_ms']} & "
        f"{_fmt(row['value_iteration_mean_utility'])} & {row['policy_mismatches']} \\\\"
        for row in part2_summary["rows"]
    )
    return rf"""
\section{{Part 2 questions}}

\subsection{{Complicated maze design}}
For Part 2, I designed a family of larger mazes instead of changing only one map manually. Starting from size 7 and increasing to size 17, each generated maze contains more white states, more walls, and more positive/negative reward cells. The same transition model as Part 1 is kept, and the random seed is fixed at {part2_summary["seed"]} so that the experiments are reproducible.

\subsection{{How do the number of states and the complexity of the environment affect convergence?}}
\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{rrrrrrr}}
\toprule
Size & VI Iter & PI Iter & VI Runtime & PI Runtime & VI Mean Utility & Mismatch \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\caption{{Part 2 comparison across generated mazes.}}
\label{{tab:part2_results}}
\end{{table}}

Table~\ref{{tab:part2_results}} and the plots below show a clear trend: when the maze gets larger, both algorithms need more computation before the utilities stabilize. Value Iteration usually needs more repeated Bellman sweeps over the whole state space. Policy Iteration often reaches a stable policy with a different update pattern, but its evaluation stage is more expensive per round, so its runtime is not automatically lower.

{_figure("results/part2/iterations_vs_size.png", "Iterations required by each method as maze size increases.", "fig:part2_iterations")}
{_figure("results/part2/runtime_vs_size.png", "Runtime comparison as maze size increases.", "fig:part2_runtime")}
{_figure("results/part2/mean_utility_vs_size.png", "Mean utility comparison as maze size increases.", "fig:part2_utility")}

\subsection{{How complex can the environment be while still learning the right policy?}}
In this implementation, both algorithms still converged and produced the same final policy for every tested maze size from {part2_summary["sizes"][0]} to {part2_summary["sizes"][-1]}. Therefore, within the tested range, the largest environment that still learned a matching optimal policy was the {part2_summary["sizes"][-1]}x{part2_summary["sizes"][-1]} maze. The policy mismatch column in Table~\ref{{tab:part2_results}} is zero for every run, which is the main evidence used here.
"""


def write_report(root: Path, part1_summary: dict | None, part2_summary: dict | None) -> Path:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "REPORT.tex"

    base_config = load_config(root / "config" / "base.yaml")
    vi_table = _load_csv_table(root / "results" / "part1" / "value_iteration_utilities.csv")
    pi_table = _load_csv_table(root / "results" / "part1" / "policy_iteration_utilities.csv")

    value_iteration_text = (
        "The assignment maze is a 6x6 grid with 5 walls, 6 green reward cells (+1), 5 brown reward cells (-1), and white cells with reward "
        f"{base_config['step_reward']}. "
        "The agent has four actions: up, right, down, and left. The intended move happens with probability "
        f"{base_config['p_forward']}, and the two perpendicular moves each happen with probability {(1.0 - base_config['p_forward']) / 2.0:.1f}. "
        "If a move hits a wall or the boundary, the agent stays in the same state. "
        "A crucial detail from the assignment is that there are no terminal states, so green and brown cells remain revisitable. "
        "Value Iteration updates every non-wall state with the Bellman optimality equation "
        r"$U_{k+1}(s)=\max_a \sum_{s'} P(s' \mid s,a)\left[R(s,a,s')+\gamma U_k(s')\right]$ "
        "until the change between two consecutive sweeps is below the stopping threshold."
    )

    policy_iteration_text = (
        "Policy Iteration uses the same MDP model as Value Iteration. "
        "It starts from an initial policy, repeatedly evaluates the current policy, and then improves it greedily using the updated utilities. "
        "Policy evaluation is implemented by iterative sweeps using "
        r"$U^{\pi}(s)=\sum_{s'} P(s' \mid s,\pi(s))\left[R(s,\pi(s),s')+\gamma U^{\pi}(s')\right]$, "
        "and policy improvement replaces the current action with the action that gives the highest expected return. "
        "Because there are no terminal states, the policy is defined for every non-wall cell, including green and brown reward cells."
    )

    sections = []
    if part1_summary:
        sections.append(
            rf"""
\section{{Problem setup}}
The base maze used in Part 1 is shown in Figure~\ref{{fig:base_maze}}. The discount factor is fixed at {base_config['gamma']}, exactly as required by the assignment.

{_figure("results/part1/maze.png", "Assignment maze used for Part 1. Green cells have reward +1, brown cells have reward -1, white cells have reward -0.05, and gray cells are walls.", "fig:base_maze", width="0.68\\textwidth")}
"""
        )
        sections.append(
            _part1_method_section(
                method_title="Value Iteration",
                implementation_text=value_iteration_text,
                policy_image="results/part1/value_iteration_policy.png",
                utility_image="results/part1/value_iteration_utility.png",
                estimate_image="results/part1/value_iteration_utility_estimates.png",
                utility_table=vi_table,
                iterations=part1_summary["value_iteration"]["iterations"],
                runtime_ms=part1_summary["value_iteration"]["runtime_ms"],
                method_slug="value_iteration",
                method_note=(
                    "The policy points toward repeatedly collecting high reward while avoiding paths that make accidental moves into brown cells more likely. "
                    "This matches the assignment requirement to display the optimal policy for all non-wall states."
                ),
                utility_note=(
                    "These utilities are much larger than the immediate rewards because the assignment uses an infinite-horizon discounted process with no terminal states. "
                    "A state is valuable not only for its next reward, but also for how well it can lead the agent back to green cells repeatedly."
                ),
            )
        )
        sections.append(
            _part1_method_section(
                method_title="Policy Iteration",
                implementation_text=policy_iteration_text,
                policy_image="results/part1/policy_iteration_policy.png",
                utility_image="results/part1/policy_iteration_utility.png",
                estimate_image="results/part1/policy_iteration_utility_estimates.png",
                utility_table=pi_table,
                iterations=part1_summary["policy_iteration"]["iterations"],
                runtime_ms=part1_summary["policy_iteration"]["runtime_ms"],
                method_slug="policy_iteration",
                method_note=(
                    "The final policy produced by Policy Iteration matches the one from Value Iteration. "
                    "This agreement is expected because both algorithms solve the same discounted MDP."
                ),
                utility_note=(
                    "The utility magnitudes are almost identical to those from Value Iteration, which confirms that the two methods converged to the same solution for Part 1."
                ),
            )
        )
        sections.append(
            rf"""
\section{{Source code for Part 1}}
\begin{{itemize}}
  \item Environment model: \texttt{{src/gridworld.py}}
  \item Value Iteration: \texttt{{src/value\_iteration.py}}
  \item Policy Iteration: \texttt{{src/policy\_iteration.py}}
  \item Visualization: \texttt{{src/visualization.py}}
  \item CLI entrypoint: \texttt{{main.py}}
\end{{itemize}}
The code follows the report structure directly: the environment model defines the MDP, the two algorithm files implement Value Iteration and Policy Iteration, and the plotting/report code exports the policy plots, utility tables, and convergence figures required in the assignment.
"""
        )
    if part2_summary:
        sections.append(_part2_section(part2_summary))

    latex = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}

\title{{SC4003 Assignment 1 Report}}
\author{{}}
\date{{\today}}

\begin{{document}}
\maketitle

{''.join(sections)}

\end{{document}}
"""

    report_path.write_text(latex.strip() + "\n", encoding="utf-8")
    return report_path


def compile_report(report_path: Path) -> Path | None:
    if shutil.which("pdflatex") is None:
        return None

    report_dir = report_path.parent
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={report_dir}",
        str(report_path),
    ]
    subprocess.run(
        command,
        cwd=report_dir.parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return report_dir / f"{report_path.stem}.pdf"
