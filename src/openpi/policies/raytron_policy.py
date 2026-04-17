import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_raytron_example() -> dict:
    """Creates a random input example for the Raytron policy."""
    return {
        "observation/head_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_arm_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_arm_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(14).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RaytronInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_rgb = _parse_image(data["observation/head_rgb"])
        left_arm_rgb = _parse_image(data["observation/left_arm_rgb"])
        right_arm_rgb = _parse_image(data["observation/right_arm_rgb"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": head_rgb,
                "left_wrist_0_rgb": left_arm_rgb,
                "right_wrist_0_rgb": right_arm_rgb,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RaytronOutputs(transforms.DataTransformFn):
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
