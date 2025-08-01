import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)

device = torch.device("mps")
models = ["prompthero/openjourney", "tencent/HunyuanWorld-1"]
pipe = StableDiffusionPipeline.from_pretrained(models[0], torch_dtype=torch.float16)
pipe = pipe.to(device)
prompt = (
    "retro series of different cars with different colors and shapes, mdjrny-v4 style"
)
image = pipe(prompt).images[0]
image.save("./outputs/retro_cars.png")
