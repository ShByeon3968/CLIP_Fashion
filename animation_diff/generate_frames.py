import torch
from diffusers import LTXPipeline, LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.utils import export_to_video, load_video
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
import uuid
import os
from abc import *

class VideoDiff(metaclass=ABCMeta):
    @abstractmethod
    def generate_animation(self):
        pass

class LTXVideoDiff(VideoDiff):
    '''
    LTX Video 13B 비디오 생성기 
    '''
    def __init__(self,device:str="cuda"):
        self.device = device
        # 파이프라인 세팅
        self.pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")

        # negative prompt 설정
        self.negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"


    def generate_animation(self,prompt:str, output_root:str="./outputs",num_frames:int=10):
        """
        Diffusion을 사용하여 애니메이션 생성
        Args:
            prompt (str): 애니메이션에 대한 설명
            output_root (str): 출력 디렉토리
            num_frames (int): 생성할 프레임 수
        """
        uid = str(uuid.uuid4())[:8]
        output_path = os.path.join(output_root, f"uid_{uid}.mp4")
        os.makedirs(output_root, exist_ok=True)
        
        # 비디오 생성
        video = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=768,
            height=512,
            num_frames=num_frames,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            num_inference_steps=50,
        ).frames[0]
        
        export_to_video(video, output_path)

    