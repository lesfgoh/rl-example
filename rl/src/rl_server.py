"""Runs the RL server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.

from fastapi import FastAPI, Request
from rl_manager import RLManager

app = FastAPI()
manager = RLManager(
    history_len=4,
    lr=2e-5,
    scout_gamma=0.75,
    guard_gamma=0.99,
    entropy_beta=0.02,
)


@app.post("/rl")
async def rl(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds an observation into the RL model.

    Returns action taken given current observation (int)
    """

    # get observation, feed into model
    input_json = await request.json()

    predictions = []
    # each is a dict with one key "observation" and the value as a dictionary observation
    for idx, instance in enumerate(input_json["instances"]):
        observation = instance["observation"]
        # add back agent id for eval
        observation["agent_id"] = f"player_{idx}"

        # reset environment on a new round
        if observation["step"] == 0:
            await reset({})

        predictions.append({"action": manager.rl(observation)})
    return {"predictions": predictions}


@app.post("/reset")
async def reset(_: Request) -> None:
    """
    Clear per-episode state of the existing manager.
    """
    # perform learning update from last episode, then reset per-episode state
    # manager.update()
    manager.reset()
    return


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
