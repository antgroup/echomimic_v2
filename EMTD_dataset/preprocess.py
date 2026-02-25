import sys

from src.utils.img_utils import pil_to_cv2, cv2_to_pil, center_crop_cv2, pils_from_video, save_videos_from_pils, save_video_from_cv2_list
from PIL import Image
import cv2
from IPython import embed
import numpy as np
import copy
from src.utils.motion_utils import motion_sync
import pathlib
import torch
import pickle
from glob import glob
import os
from src.models.dwpose.dwpose_detector import dwpose_detector as dwprocessor
from src.models.dwpose.util import draw_pose
import decord
from tqdm import tqdm
from moviepy.editor import AudioFileClip, VideoFileClip
from multiprocessing.pool import ThreadPool

##################################
base_dir = "root"
tasks = ["emtd"]

process_num = 800 #1266

start = 0
end = process_num + start
#################################
MAX_SIZE = 768


def convert_fps(src_path, tgt_path, tgt_fps=24, tgt_sr=16000):
    clip = VideoFileClip(src_path)
    new_clip = clip.set_fps(tgt_fps)
    if tgt_fps is not None:
        audio = new_clip.audio
        audio = audio.set_fps(tgt_sr)
        new_clip = new_clip.set_audio(audio)
    
    new_clip.write_videofile(tgt_path, codec='libx264', audio_codec='aac')
    
def get_video_pose(
        video_path: str, 
        sample_stride: int=1,
        max_frame=None):

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    if max_frame is not None:
        frames = frames[0:max_frame,:,:]
    height, width, _ = frames[0].shape
    # detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    detected_poses = [dwprocessor(frm) for frm in frames]
    dwprocessor.release_memory()

    return detected_poses, height, width, frames
    
def resize_and_pad(img, max_size):
    img_new = np.zeros((max_size, max_size, 3)).astype('uint8')
    imh, imw = img.shape[0], img.shape[1]
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new

    img_resize = cv2.resize(img, (imw_new, imh_new))
    img_new[rb:re,cb:ce,:] = img_resize
    return img_new

def resize_and_pad_param(imh, imw, max_size):
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        imh_new = max_size

        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new
        
    return imh_new, imw_new, rb, re, cb, ce

def get_pose_params(detected_poses, max_size):
    print('get_pose_params...')
    # pose rescale 
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]['num'] = num
        candidate_body = detected_pose['bodies']['candidate']
        score_body = detected_pose['bodies']['score']
        candidate_face = detected_pose['faces']
        score_face = detected_pose['faces_score']
        candidate_hand = detected_pose['hands']
        score_hand = detected_pose['hands_score']

        # 选取置信度最高的face
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
            detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]

        # 选取置信度最高的body
        if score_body.shape[0] > 1:
            tmp_score = []
            for k in range(0, score_body.shape[0]):
                tmp_score.append(score_body[k].mean())
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index*18:(index+1)*18,:]
            score_body = score_body[index]
            score_hand = score_hand[(index*2):(index*2+2),:]
            candidate_hand = candidate_hand[(index*2):(index*2+2),:,:]
        else:
            score_body = score_body[0]
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score>0.8]


        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]


        face_pose = candidate_face
        hand_pose = candidate_hand


        h_min, h_max = np.min(face_pose[:,1]), np.max(body_pose[:7,1])

        h_ = h_max - h_min
        
        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2
        
        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)

    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)
    print(mid)

    margin_ratio = 0.25
    h_margin = (h_max-h_min)*margin_ratio
    
    h_min = max(h_min-h_margin*0.65, 0)
    h_max = min(h_max+h_margin*0.5, 1)

    h_new = h_max - h_min
    
    h_min_real = int(h_min*height)
    h_max_real = int(h_max*height)
    mid_real = int(mid*width)
    
    
    height_new = h_max_real-h_min_real+1
    width_new = height_new
    w_min_real = mid_real - height_new // 2

    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width

    print(width_new, height_new)

    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(height_new, width_new, max_size)
    res = {'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce], 
           'pose_params': [w_min, w_max, h_min, h_max],
           'video_params': [h_min_real, h_max_real, w_min_real, w_max_real],
           }
    return res

def save_pose_params_item(input_items):
    detected_pose, pose_params, draw_pose_params, save_dir = input_items
    w_min, w_max, h_min, h_max = pose_params
    num = detected_pose['num']
    candidate_body = detected_pose['bodies']['candidate']
    candidate_face = detected_pose['faces'][0]
    candidate_hand = detected_pose['hands']
    candidate_body[:,0] = (candidate_body[:,0]-w_min)/(w_max-w_min)
    candidate_body[:,1] = (candidate_body[:,1]-h_min)/(h_max-h_min)
    candidate_face[:,0] = (candidate_face[:,0]-w_min)/(w_max-w_min)
    candidate_face[:,1] = (candidate_face[:,1]-h_min)/(h_max-h_min)
    candidate_hand[:,:,0] = (candidate_hand[:,:,0]-w_min)/(w_max-w_min)
    candidate_hand[:,:,1] = (candidate_hand[:,:,1]-h_min)/(h_max-h_min)
    detected_pose['bodies']['candidate'] = candidate_body
    detected_pose['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
    detected_pose['hands'] = candidate_hand
    detected_pose['draw_pose_params'] = draw_pose_params
    np.save(save_dir+'/'+str(num)+'.npy', detected_pose)

def save_pose_params(detected_poses, pose_params, draw_pose_params, ori_video_path):
    save_dir = ori_video_path.replace('original_videos', 'image_audio_features/pose/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_list = []
    for i, detected_pose in enumerate(detected_poses):
        input_list.append([detected_pose, pose_params, draw_pose_params, save_dir])

    pool = ThreadPool(8)
    pool.map(save_pose_params_item, input_list)
    pool.close()
    pool.join()
       
def save_processed_video(ori_frames, video_params, ori_video_path, max_size):
    save_path = ori_video_path.replace('original_videos', 'processed/video/')
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    h_min_real, h_max_real, w_min_real, w_max_real = video_params
    video_frame_crop = []
    for img in ori_frames:
        img = img[h_min_real:h_max_real,w_min_real:w_max_real,:]
        img = resize_and_pad(img, max_size=max_size)
        video_frame_crop.append(img)
    save_video_from_cv2_list(video_frame_crop, save_path, fps=24.0, rgb2bgr=True)
    return video_frame_crop

def save_audio(ori_video_path, sub_task):
    save_path = ori_video_path.replace('original_videos', 'processed/audio/')
    save_dir = os.path.dirname(save_path)
    save_path = save_path + '.wav'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ori_video_path = ori_video_path.replace(sub_task, sub_task+'_24fps')
    audio_clip = AudioFileClip(ori_video_path)
    audio_clip.write_audiofile(save_path)

def draw_pose_video(pose_params_path, save_path, max_size, ori_frames=None):
    pose_files = os.listdir(pose_params_path)
     # 生成Pose图cd pro 
    output_pose_img = []
    for i in range(0, len(pose_files)):
        pose_params_path_tmp = pose_params_path + '/' + str(i) + '.npy'
        detected_pose = np.load(pose_params_path_tmp, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
        im = draw_pose(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im),(1,2,0))
        img_new = np.zeros((max_size, max_size, 3)).astype('uint8')
        img_new[rb:re,cb:ce,:] = im
        if ori_frames is not None:
            img_new = img_new * 0.6 + ori_frames[i] * 0.4
            img_new = img_new.astype('uint8')
        output_pose_img.append(img_new)

    output_pose_img = np.stack(output_pose_img)
    save_video_from_cv2_list(output_pose_img, save_path, fps=24.0, rgb2bgr=True)
    print('save to ' + save_path)

visualization = False
for sub_task in tasks:

    ori_list = os.listdir(base_dir+sub_task)[start:end]
    
    mp4_list = ori_list

    new_dir = base_dir+sub_task+'_24fps'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    index = 1
    for i, mp4_file in enumerate(mp4_list):
        ori_video_path = base_dir+sub_task+'/'+mp4_file
        if ori_video_path[-3:]=='mp4' or ori_video_path[-3:] =='MOV':
            try:
                # 转换祯率
                ori_video_path_new = ori_video_path.replace(sub_task, sub_task+'_24fps')
                if '.MOV' in ori_video_path_new:
                    ori_video_path_new.replace('.MOV', '.mp4')
                convert_fps(ori_video_path, ori_video_path_new)
                print([index+start, ori_video_path, start, end])
                # 提取Pose
                detected_poses, height, width, ori_frames = get_video_pose(ori_video_path_new, max_frame=None)
                print(height, width)
                # 提取相关参数
                res_params = get_pose_params(detected_poses, MAX_SIZE)
                # 存储Pose参数
                save_pose_params(detected_poses, res_params['pose_params'], res_params['draw_pose_params'], ori_video_path)
                # 存储截取视频
                video_frame_crop = save_processed_video(ori_frames, res_params['video_params'], ori_video_path, MAX_SIZE)
                # 存储音频
                save_audio(ori_video_path, sub_task)
                index += 1
                if visualization:
                    # 绘制pose图
                    pose_params_path = ori_video_path.replace('original_videos', 'image_audio_features/pose')
                    save_path = "./vis_pose_results/" + os.path.basename(ori_video_path)
                    draw_pose_video(pose_params_path, save_path, ori_frames=video_frame_crop)
            except Exception:
                print(["extract crash!", index+start, ori_video_path, start, end])
                continue 

    print(["All Finished", sub_task, start, end])
