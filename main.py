from inspect_ai import eval as inspect_eval

from analyse import main as analyse_results
from rugby_best import rugby_best

MODEL = "anthropic/claude-haiku-4-5"


def main() -> None:
    inspect_eval(rugby_best, model=MODEL)
    analyse_results()


if __name__ == "__main__":
    main()
