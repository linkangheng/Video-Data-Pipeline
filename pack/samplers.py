import os
import cv2
import numpy as np
from PIL import Image
import sys 
sys.path.append("/data/")
from tools import extract_frames, get_cache_video, uniform_sample, get_video_total_frames, keyframes_sampler, combineKeyFrames, load_image
from dataset_loader import get_prefix
from dataset import videoItem
import subprocess
import ffmpeg

def get_image(file_idx, image_path, args=None):
    image_name_list = []
    image_dict_list = []
    img = load_image(image_path)
    image_name_list.append(f"{file_idx:09d}")
    image_dict_list.append(dict(
        __key__=f"{file_idx:09d}",
        jpg=img,
    ))
    return image_name_list, image_dict_list

def get_images(file_idx, image_paths):
    image_name_list = []
    image_dict_list = []
    for index, image_path in enumerate(image_paths):
        img = load_image(image_path)
        image_name_list.append(f"{file_idx:09d}-{index}")
        image_dict_list.append(dict(
            __key__=f"{file_idx:09d}-{index}",
            jpg=img,
        ))
    return image_name_list, image_dict_list

def get_unicontrol_images(file_idx, source, target):
    image_name_list = []
    image_dict_list = []
    for index, image_path in enumerate([source, target]):
        img = load_image(image_path, type="bytes")
        image_name_list.append(f"{file_idx:09d}-{index}")
        image_dict_list.append(dict(
            __key__=f"{file_idx:09d}-{index}",
            jpg=img,
        ))
    return image_name_list, image_dict_list

def Merlin_S_sampler(file_idx, image_paths, args=None):
    image_name_list = []
    image_dict_list = []

    for index, path in enumerate(image_paths):
        image_path = path['image_name']
        if image_path.startswith("Black background"):
            size = [int(i.replace(" ","")) for i in image_path.split(":")[-1].split(", ")]
            w = size[0]
            h = size[1]
            img = np.zeros((h,w,3), np.uint8)
        else:
            if "data//" in path['image_name']:
                image = path['image_name'].replace("data//", "data/")
            else:
                image = path['image_name']
            img = load_image(image)
        image_name_list.append(f"{file_idx:09d}-{index}")
        image_dict_list.append(dict(
            __key__=f"{file_idx:09d}-{index}",
            jpg=img,
        ))

    return image_name_list, image_dict_list

def getVideoList(file_idx, video_paths, args=None):

    video_name_list = []
    video_dict_list = []

    if not isinstance(video_paths, list):
        video_paths = [video_paths]

    for video_path in video_paths:
        if video_path.startswith("s3://"):
            remote_path = video_path
            video_path = get_cache_video(remote_path)
        else:
            remote_path = None
        
        with open(video_path, 'rb') as f:
            video = f.read()
        
        if remote_path:
            os.remove(video_path)

        video_name_list.append(f"{file_idx:09d}")
        video_dict_list.append(dict(
            __key__=f"{file_idx:09d}",
            mp4=video,
        ))
    
    return video_name_list, video_dict_list

def uniformSampler(file_idx, video_path, args=None):
    # 用于均匀采样13帧
    image_name_list = []
    image_dict_list = []

    # video_prefix = get_prefix(args.dataset)
    # video_path = os.path.join(video_prefix, video_path)
    video = videoItem(video_path, 'oss')    
    images = video.read_video(num_segments=16)

    for index, img in enumerate(images):
        image_name_list.append(f"{file_idx:09d}-{index}")
        image_dict_list.append(dict(
            __key__=f"{file_idx:09d}-{index}",
            jpg=img,
        ))
    
    return image_name_list, image_dict_list

def keyFrameSampler(file_idx, video_path, args=None):
    #  Only supported for InternVid dataset, whose video is local; I_total_frames == P_total_frames below
    video_prefix = get_prefix(args.dataset)
    video_path = os.path.join(video_prefix, video_path)
    
    if video_path.startswith("s3://"):
        remote_path = video_path
        video_path = get_cache_video(remote_path)
    else:
        remote_path = ""
    
    I_images,I_indices,I_total_frames = keyframes_sampler(video_path, 'I', max_samples = args.Iframes) 
    len_PFrames = args.total_frames - len(I_images)
    P_images,P_indices,P_total_frames = keyframes_sampler(video_path, 'P', max_samples = len_PFrames)

    # assert len(I_images) == len(I_indices), "len(I_images) != len(I_indices)"
    # assert len(P_images) == len(P_indices), "len(P_images) != len(P_indices)"
    assert I_total_frames == P_total_frames, "I_total_frames != P_total_frames"
    try:
        images, indices, frame_types = combineKeyFrames(I_images, I_indices, P_images, P_indices)
    except:
        print(f"I_images: {len(I_images)}, I_indices: {len(I_indices)}, P_images: {len(P_images)}, P_indices: {len(P_indices)}")
    indices_list = [int((i/(I_total_frames-1)) * args.time_scale) for i in indices]
    image_name_list = []
    image_dict_list = []
    
    for index, img in enumerate(images):
        image_name_list.append(f"{file_idx:09d}-{index}")
        image_dict_list.append(dict(
            __key__=f"{file_idx:09d}-{index}",
            jpg=img,
        ))
    
    if len(remote_path) > 0:
        os.remove(video_path)
    
    return image_name_list, image_dict_list, indices_list, frame_types

def Momentor(file_idx, video_path, args=None):
    video_prefix = get_prefix(args.dataset)
    
    pass

def debug():
    class args:
        Iframes = 8
        max_frames = 24
        time_scale = 1000
        dataset = "internvid"
        total_frames = 24

    file_idx = 0
    video_path = "/data/webvid/debug/test.mp4"
    KF_sampler(file_idx, video_path, args=args)


