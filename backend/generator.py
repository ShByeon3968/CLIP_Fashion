# generator.py
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel,StableDiffusion3Pipeline,StableDiffusion3ControlNetPipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
import torch
from PIL import Image
from controlnet_preprocess import get_canny_map
from prompt_suggester import *

def generate_redesign(input_image: Image.Image, user_prompt: str):

    # 전처리 수행
    control_image = get_canny_map(input_image)

    # # 모델 로딩
    # controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     controlnet=controlnet,
    #     torch_dtype=torch.float16
    # ).to("cuda")
    # pipe.enable_xformers_memory_efficient_attention()

    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet,torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    # pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    # 생성
    output = pipe(prompt=user_prompt,negative_prompt="low quality, worst quality, blurry",
                  control_image=control_image, num_inference_steps=30)
    return output.images[0]
