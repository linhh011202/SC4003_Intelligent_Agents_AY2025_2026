from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .gridworld import GridWorld, State
from .utils import ACTION_TO_GLYPH


CELL = 72
PADDING = 40
TITLE_HEIGHT = 40


def _lerp_channel(low: int, high: int, ratio: float) -> int:
    return int(low + (high - low) * ratio)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _interpolate_color(low: tuple[int, int, int], high: tuple[int, int, int], ratio: float) -> str:
    ratio = _clamp(ratio, 0.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(
        _lerp_channel(low[0], high[0], ratio),
        _lerp_channel(low[1], high[1], ratio),
        _lerp_channel(low[2], high[2], ratio),
    )


def _board_dimensions(env: GridWorld) -> tuple[int, int]:
    width = env.cols * CELL + PADDING * 2
    height = env.rows * CELL + PADDING * 2 + TITLE_HEIGHT
    return width, height


def _cell_origin(row: int, col: int) -> tuple[int, int]:
    x = PADDING + col * CELL
    y = PADDING + TITLE_HEIGHT + row * CELL
    return x, y


def _cell_base_color(env: GridWorld, state: State) -> str:
    if env.is_wall(state):
        return "#666666"
    if state in env.rewards:
        return "#82d173" if env.rewards[state] > 0 else "#f4a259"
    return "#f3f3f3"


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="monospace" font-size="18" fill="#111111">{escape(title)}</text>',
    ]


def _draw_base_board(env: GridWorld, title: str) -> list[str]:
    width, height = _board_dimensions(env)
    parts = _svg_header(width, height, title)

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            x, y = _cell_origin(row, col)
            fill = _cell_base_color(env, state)
            stroke = "#1f1f1f"
            stroke_width = 1
            if env.start_state == state:
                stroke = "#2d6cdf"
                stroke_width = 3

            parts.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )

            if env.is_wall(state):
                parts.append(
                    f'<text x="{x + CELL / 2:.1f}" y="{y + CELL / 2 + 8:.1f}" text-anchor="middle" font-family="monospace" font-size="18" fill="#111111">WALL</text>'
                )
            elif state in env.rewards:
                parts.append(
                    f'<text x="{x + CELL / 2:.1f}" y="{y + CELL / 2 + 8:.1f}" text-anchor="middle" font-family="monospace" font-size="18" fill="#111111">{env.rewards[state]:+.2f}</text>'
                )
    return parts


def _write_svg(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _setup_axes(env: GridWorld, title: str):
    fig_width = max(6.0, env.cols * 1.2)
    fig_height = max(5.0, env.rows * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(env.rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols + 1))
    ax.set_yticks(range(env.rows + 1))
    ax.grid(color="#1f1f1f", linewidth=1)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig, ax


def _draw_board(ax, env: GridWorld, annotate_rewards: bool = True) -> None:
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            color = _cell_base_color(env, state)
            rect = Rectangle((col, row), 1, 1, facecolor=color, edgecolor="#1f1f1f")
            ax.add_patch(rect)
            if env.start_state == state:
                border = Rectangle((col, row), 1, 1, fill=False, edgecolor="#2d6cdf", linewidth=3)
                ax.add_patch(border)
            if env.is_wall(state):
                ax.text(col + 0.5, row + 0.55, "WALL", ha="center", va="center", fontsize=10, fontweight="bold")
            elif annotate_rewards and state in env.rewards:
                ax.text(col + 0.5, row + 0.55, f"{env.rewards[state]:+.1f}", ha="center", va="center", fontsize=11, fontweight="bold")


def _save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_board_png(env: GridWorld, out_path: Path, title: str) -> None:
    fig, ax = _setup_axes(env, title)
    _draw_board(ax, env, annotate_rewards=True)
    _save_figure(fig, out_path)


def save_policy_png(
    env: GridWorld,
    policy: dict[State, str | None],
    utilities: dict[State, float],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = _setup_axes(env, title)
    _draw_board(ax, env, annotate_rewards=False)
    for state, action in policy.items():
        if action is None or env.is_terminal(state):
            continue
        row, col = state
        ax.text(col + 0.5, row + 0.43, ACTION_TO_GLYPH[action], ha="center", va="center", fontsize=20, fontweight="bold")
        if state in env.rewards:
            ax.text(col + 0.18, row + 0.18, f"R={env.rewards[state]:+.0f}", ha="left", va="top", fontsize=7, fontweight="bold")
        ax.text(col + 0.5, row + 0.78, f"{utilities[state]:.3f}", ha="center", va="center", fontsize=8)
    _save_figure(fig, out_path)


def save_utility_heatmap_png(
    env: GridWorld,
    utilities: dict[State, float],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = _setup_axes(env, title)
    values = [utilities[state] for state in env.get_states() if not env.is_wall(state)]
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if env.is_wall(state):
                facecolor = "#666666"
                label = "W"
            else:
                ratio = 0.5 if high == low else (utilities[state] - low) / (high - low)
                facecolor = _interpolate_color((244, 162, 89), (130, 209, 115), ratio)
                label = f"{utilities[state]:.3f}"
            rect = Rectangle((col, row), 1, 1, facecolor=facecolor, edgecolor="#1f1f1f")
            ax.add_patch(rect)
            ax.text(col + 0.5, row + 0.52, label, ha="center", va="center", fontsize=9, fontweight="bold")
            if state in env.rewards and not env.is_wall(state):
                ax.text(col + 0.16, row + 0.16, f"R={env.rewards[state]:+.0f}", ha="left", va="top", fontsize=7, fontweight="bold")
    _save_figure(fig, out_path)


def plot_series_png(
    series: dict[str, list[tuple[float, float]]],
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    for index, (label, points) in enumerate(series.items()):
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(
            xs,
            ys,
            linewidth=1.5,
            label=label,
            color=palette[index % len(palette)],
            alpha=0.9,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        ncol=2,
        frameon=False,
    )
    _save_figure(fig, out_path)


def save_board_svg(env: GridWorld, out_path: Path, title: str) -> None:
    parts = _draw_base_board(env, title)
    _write_svg(out_path, parts)


def save_policy_svg(
    env: GridWorld,
    policy: dict[State, str | None],
    utilities: dict[State, float],
    out_path: Path,
    title: str,
) -> None:
    parts = _draw_base_board(env, title)
    for state, action in policy.items():
        if action is None or env.is_terminal(state):
            continue
        row, col = state
        x, y = _cell_origin(row, col)
        glyph = ACTION_TO_GLYPH[action]
        parts.append(
            f'<text x="{x + CELL / 2:.1f}" y="{y + CELL / 2 - 4:.1f}" text-anchor="middle" font-family="monospace" font-size="28" fill="#111111">{glyph}</text>'
        )
        parts.append(
            f'<text x="{x + CELL / 2:.1f}" y="{y + CELL - 10:.1f}" text-anchor="middle" font-family="monospace" font-size="12" fill="#333333">{utilities[state]:.3f}</text>'
        )
    _write_svg(out_path, parts)


def save_utility_heatmap_svg(
    env: GridWorld,
    utilities: dict[State, float],
    out_path: Path,
    title: str,
) -> None:
    values = [utilities[state] for state in env.get_non_terminal_states()]
    if values:
        low = min(values)
        high = max(values)
    else:
        low = 0.0
        high = 1.0

    width, height = _board_dimensions(env)
    parts = _svg_header(width, height, title)
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            x, y = _cell_origin(row, col)
            if env.is_wall(state):
                fill = "#666666"
                label = "WALL"
            elif env.is_terminal(state):
                fill = "#dedede"
                label = f"T {env.rewards.get(state, 0.0):+.2f}"
            else:
                if high == low:
                    ratio = 0.5
                else:
                    ratio = (utilities[state] - low) / (high - low)
                fill = _interpolate_color((244, 162, 89), (130, 209, 115), ratio)
                label = f"{utilities[state]:.3f}"

            parts.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="#1f1f1f" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + CELL / 2:.1f}" y="{y + CELL / 2 + 8:.1f}" text-anchor="middle" font-family="monospace" font-size="13" fill="#111111">{label}</text>'
            )
    _write_svg(out_path, parts)


def plot_series_svg(
    series: dict[str, list[tuple[float, float]]],
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width = 960
    height = 540
    left = 80
    right = 40
    top = 60
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [point for points in series.values() for point in points]
    x_values = [point[0] for point in all_points] or [0.0, 1.0]
    y_values = [point[1] for point in all_points] or [0.0, 1.0]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    colors = ["#2d6cdf", "#dd6b20", "#2f855a", "#c53030"]
    parts = _svg_header(width, height, title)
    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="monospace" font-size="14" fill="#111111">{escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="20" y="{height / 2:.1f}" text-anchor="middle" font-family="monospace" font-size="14" fill="#111111" transform="rotate(-90 20 {height / 2:.1f})">{escape(y_label)}</text>'
    )

    for tick in range(6):
        x_ratio = tick / 5.0
        y_ratio = tick / 5.0
        x = left + plot_width * x_ratio
        y = top + plot_height * (1.0 - y_ratio)
        x_value = min_x + (max_x - min_x) * x_ratio
        y_value = min_y + (max_y - min_y) * y_ratio

        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#333333" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="monospace" font-size="12" fill="#333333">{x_value:.2f}</text>'
        )
        parts.append(
            f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333333" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="monospace" font-size="12" fill="#333333">{y_value:.3f}</text>'
        )

    legend_x = left + 10
    legend_y = 40
    for index, (label, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        mapped = []
        for x_value, y_value in points:
            x = left + ((x_value - min_x) / (max_x - min_x)) * plot_width
            y = top + (1.0 - ((y_value - min_y) / (max_y - min_y))) * plot_height
            mapped.append(f"{x:.1f},{y:.1f}")
        if mapped:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(mapped)}"/>'
            )
        ly = legend_y + index * 20
        parts.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 20}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 28}" y="{ly + 4}" font-family="monospace" font-size="12" fill="#111111">{escape(label)}</text>'
        )

    _write_svg(out_path, parts)
