from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log


def latest_log() -> str:
    """Return the path to the most recent .eval log in logs/."""
    logs = sorted(Path("logs").glob("*.eval"))
    if not logs:
        raise FileNotFoundError("No .eval logs found in logs/")
    return str(logs[-1])


def normalise(answer: str) -> str:
    """Tidy a raw model answer into a comparable name."""
    return answer.strip().strip(".").strip()


def main() -> None:
    log = read_eval_log(latest_log())
    answers = [normalise(s.output.completion) for s in log.samples]
    counts = Counter(answers)

    print(f"{len(answers)} runs, {len(counts)} distinct answers:")
    for name, n in counts.most_common():
        print(f"  {n:3d}  {name}")

    names = [name for name, _ in counts.most_common()]
    values = [n for _, n in counts.most_common()]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Count")
    plt.title("'Greatest rugby player of all time?' - answer distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("distribution.png", dpi=150)
    print("Saved distribution.png")


if __name__ == "__main__":
    main()
