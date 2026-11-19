import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
import sys

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2

from decord import VideoReader
from moviepy.editor import VideoFileClip, AudioFileClip

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/infer.yaml")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("-L", type=int, default=240)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ref_images_dir", type=str, default=f'./assets/halfbody_demo/refimag')
    parser.add_argument("--audio_dir", type=str, default='./assets/halfbody_demo/audio')
    parser.add_argument("--pose_dir", type=str, default="./assets/halfbody_demo/pose")
    parser.add_argument("--refimg_name", type=str, default='natural_bk_openhand/0035.png')
    parser.add_argument("--audio_name", type=str, default='chinese/echomimicv2_woman.wav')
    parser.add_argument("--pose_name", type=str, default="01")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    model_flag = '{}-iter{}'.format(config.motion_module_path.split('/')[-2], config.motion_module_path.split('/')[-1].split('-')[-1][:-4])
    save_dir = Path(f"outputs/{model_flag}-seed{args.seed}/")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(save_dir)

    ############# model_init started #############
    ## vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(device, dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    ## denoising net init
    if os.path.exists(config.motion_module_path):
        print('using motion module')
    else:
        exit("motion module not found")
        ### stage1 + stage2
    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
   
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )

    # pose net init
    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )
    pose_net.load_state_dict(torch.load(config.pose_encoder_path))

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)
   
    ############# model_init finished #############
    width, height = args.W, args.H
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    if args.seed is not None and args.seed > -1:
        generator = torch.manual_seed(args.seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))

    final_fps = args.fps
        
    ref_images_dir = args.ref_images_dir
    audio_dir = args.audio_dir
    pose_dir = args.pose_dir

    refimg_name = args.refimg_name
    audio_name = args.audio_name
    pose_name = args.pose_name
    

    inputs_dict = {
        "refimg": f'{ref_images_dir}/{refimg_name}',
        "audio": f'{audio_dir}/{audio_name}',
        "pose": f'{pose_dir}/{pose_name}',
    }

    start_idx = 0

    print('Pose:', inputs_dict['pose'])
    print('Reference:', inputs_dict['refimg'])
    print('Audio:', inputs_dict['audio'])


    reference_path = Path(refimg_name)
    ref_flag = '.'.join(filter(None, (reference_path.parent.name, reference_path.name)))

    save_path = Path(f"{save_dir}/{ref_flag}/{pose_name}")
    
    save_path.mkdir(exist_ok=True, parents=True)
    ref_s = reference_path.stem
    save_name = f"{save_path}/{ref_s}-a-{audio_name}-i{start_idx}"
    
    ref_image_pil = Image.open(inputs_dict['refimg']).resize((args.W, args.H))
    audio_clip = AudioFileClip(inputs_dict['audio'])
    
    args.L = min(args.L, int(audio_clip.duration * final_fps), len(os.listdir(inputs_dict['pose'])))

    pose_list = []
    for index in range(start_idx, start_idx + args.L):
        tgt_musk = np.zeros((args.W, args.H, 3)).astype('uint8')
        tgt_musk_path = os.path.join(inputs_dict['pose'], "{}.npy".format(index))
        detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im),(1, 2, 0))
        tgt_musk[rb:re,cb:ce,:] = im

        tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
        pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2,0,1) / 255.0)
    
    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    audio_clip = AudioFileClip(inputs_dict['audio'])
    
    audio_clip = audio_clip.set_duration(args.L / final_fps)
    video = pipe(
        ref_image_pil,
        inputs_dict['audio'],
        poses_tensor[:,:,:args.L,...],
        width,
        height,
        args.L,
        args.steps,
        args.cfg,
        generator=generator,
        audio_sample_rate=args.sample_rate,
        context_frames=args.context_frames,
        fps=final_fps,
        context_overlap=args.context_overlap,
        start_idx=start_idx,
    ).videos 
    
    final_length = min(video.shape[2], poses_tensor.shape[2], args.L)
    video_sig = video[:, :, :final_length, :, :]
    
    save_videos_grid(
        video_sig,
        save_name + "_woa_sig.mp4",
        n_rows=1,
        fps=final_fps,
    )

    video_clip_sig = VideoFileClip(save_name + "_woa_sig.mp4",)
    video_clip_sig = video_clip_sig.set_audio(audio_clip)
    video_clip_sig.write_videofile(save_name + "_sig.mp4", codec="libx264", audio_codec="aac", threads=2)
    os.system("rm {}".format(save_name + "_woa_sig.mp4"))
    print(save_name)


if __name__ == "__main__":
    main()
