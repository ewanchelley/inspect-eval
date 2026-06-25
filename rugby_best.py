from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

NEUTRAL_PROMPT = (
    "Who is the greatest rugby player of all time? "
    "Reply with only the player's full name, nothing else."
)


@task
def rugby_best():
    return Task(
        dataset=[Sample(input=NEUTRAL_PROMPT)],
        solver=generate(),
        config=GenerateConfig(temperature=1.0),
        epochs=Epochs(30),
    )
