from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval

from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def capitals():
    return Task(
        dataset=[
            Sample(input="What is the capital of France?", target="Paris"),
            Sample(input="What is the capital of Japan?", target="Tokyo"),
        ],
        solver=generate(),
        scorer=includes(),
    )

def main():
    inspect_eval(capitals, model="anthropic/claude-haiku-4-5")

if __name__ == "__main__":
    main()
