from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import EvalLog, read_eval_log

COLOR = tuple[float, float, float, float]


def latest_log() -> str:
    """Return the path to the most recent .eval log in logs/."""
    logs = sorted(Path("logs").glob("*.eval"))
    if not logs:
        raise FileNotFoundError("No .eval logs found in logs/")
    return str(logs[-1])


def normalise(answer: str) -> str:
    """Tidy a raw model answer into a comparable name."""
    return answer.strip().strip(".").strip()


def group_by_framing(log: EvalLog) -> dict[str, list[str]]:
    """Collect each run's answer, keyed by its framing label."""
    by_framing: dict[str, list[str]] = defaultdict(list)
    for sample in log.samples or []:
        framing = sample.metadata["framing"]
        by_framing[framing].append(normalise(sample.output.completion))
    return by_framing


def ordered_players(by_framing: dict[str, list[str]]) -> list[str]:
    """All players, ordered by overall frequency across every framing."""
    overall: Counter[str] = Counter()
    for answers in by_framing.values():
        overall.update(answers)
    return [name for name, _ in overall.most_common()]


def assign_colours(players: list[str]) -> dict[str, COLOR]:
    """Give each player a stable colour, shared across all charts."""
    palette = plt.get_cmap("tab10")
    return {name: palette(i) for i, name in enumerate(players)}


def print_summaries(by_framing: dict[str, list[str]]) -> None:
    """Print the answer tally for each framing."""
    for framing, answers in by_framing.items():
        counts = Counter(answers)
        print(f"\n[{framing}] {len(answers)} runs, {len(counts)} distinct answers:")
        for name, n in counts.most_common():
            print(f"  {n:3d}  {name}")


def plot_distributions(
    by_framing: dict[str, list[str]],
    players: list[str],
    colours: dict[str, COLOR],
) -> None:
    """Draw one bar chart per framing, sharing player order and colours."""
    framings = list(by_framing)
    fig, axes = plt.subplots(
        1, len(framings), figsize=(5 * len(framings), 5), sharey=True
    )

    for ax, framing in zip(axes, framings):
        counts = Counter(by_framing[framing])
        # Same player order in every chart; absent players show a zero bar.
        values = [counts.get(name, 0) for name in players]
        bar_colours = [colours[name] for name in players]
        ax.bar(players, values, color=bar_colours)
        ax.set_title(framing)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("'Greatest rugby player?' - answer distribution by framing")
    fig.supylabel("Count")
    fig.tight_layout()
    fig.savefig("framing_comparison.png", dpi=150)
    print("\nSaved framing_comparison.png")


def main() -> None:
    log = read_eval_log(latest_log())
    by_framing = group_by_framing(log)
    print_summaries(by_framing)

    players = ordered_players(by_framing)
    colours = assign_colours(players)
    plot_distributions(by_framing, players, colours)


if __name__ == "__main__":
    main()
