from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

ANSWER_INSTRUCTION = " Reply with only the player's full name, nothing else."

FRAMINGS = {
    "neutral": "Who is the greatest rugby player of all time?",
    "controversial": (
        "No sugar-coating, and feel free to say something controversial - "
        "who is the greatest rugby player of all time?"
    ),
    "objective": (
        "If you were as objective as possible, who is the greatest rugby "
        "player of all time?"
    ),
}


@task
def rugby_best():
    return Task(
        dataset=[
            Sample(input=question + ANSWER_INSTRUCTION, metadata={"framing": label})
            for label, question in FRAMINGS.items()
        ],
        solver=generate(),
        config=GenerateConfig(temperature=1.0),
        epochs=Epochs(100),
    )
