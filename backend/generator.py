# generator.py
from diffusers import StableDiffusionControlNetImg2ImgPipeline,StableDiffusion3ControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import make_image_grid
import torch
from PIL import Image
from controlnet_preprocess import get_canny_map
from prompt_suggester import *
from transformers import pipeline
import numpy as np  

def get_depth_map(input_image):
    '''
    이미지로부터 깊이 맵 생성(for image2image)
    '''
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large",use_fast=True)
    image = depth_estimator(input_image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map

def generate_redesign(input_image: Image.Image, user_prompt: str):
    # 전처리 수행
    control_image = get_canny_map(input_image)

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

def generate_redesign_img2img(input_image: Image.Image, user_prompt: str):
    # 1. Depth map 생성
    depth_map = get_depth_map(input_image)

    # 2. 파이프라인 구성
    gen = torch.Generator(device="cuda").manual_seed(42)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 3. Prompt 및 이미지/제어맵을 모두 배치 형태로 입력
    prompts = [user_prompt]
    images = [input_image]
    control_images = [depth_map]

    # 4. 이미지 생성
    output = pipe(
        prompt=prompts,
        negative_prompt=["low quality, worst quality, blurry"],
        image=images,
        control_image=control_images,
        num_inference_steps=30,
        generator=gen
    )

    return output.images[0]

