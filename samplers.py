import os
import cv2
import numpy as np
from PIL import Image
from tools import extract_frames, get_cache_video, uniform_sample, get_video_total_frames, keyframes_sampler, combineKeyFrames
import subprocess



def Un_sampler(file_idx, video_path, args=None):
    # 用于均匀采样13帧
    image_name_list = []
    image_dict_list = []
    
    if args.dataset == 'webvid':
        video_prefix = "s3://vision-language-data/video-data/webvid10m/process_videos/"
    elif args.dataset == 'hd3m':
        video_prefix = "s3://vision-language-data/video-data/hd130m/process_videos/"
    elif args.dataset == 'internvid':
        video_prefix = "/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/InternVId-FLT/"
        
    video_path = os.path.join(video_prefix, video_path)
    
    if video_path.startswith("s3://"):
        remote_path = video_path
        video_path = get_cache_video(remote_path)
    else:
        remote_path = None
    
    images = extract_frames(video_path)
    
    if remote_path:
        os.remove(video_path)

    for index, img in enumerate(images):
        image_name_list.append(f"{file_idx:09d}-{index}")
        image_dict_list.append(dict(
            __key__=f"{file_idx:09d}-{index}",
            jpg=img,
        ))
    
    return image_name_list, image_dict_list

def KF_sampler(file_idx, video_path, args=None):
    #  Only supported for InternVid dataset, whose video is local; I_total_frames == P_total_frames below
    if args.dataset == 'webvid':
        video_prefix = "s3://vision-language-data/video-data/webvid10m/process_videos/"
    elif args.dataset == 'hd3m':
        video_prefix = "s3://vision-language-data/video-data/hd130m/process_videos/"
    elif args.dataset == 'internvid':
        video_prefix = "/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/InternVId-FLT/"
    else:
        raise NotImplementedError("process_dataset not supported")
    
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