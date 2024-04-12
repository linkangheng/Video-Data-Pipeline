# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Megvii, Inc. All rights reserved.
# ------------------------------------------------------------------------------------------------
import os
import re
# import fire
import megfile
import glob
import argparse
import pickle
import json
import csv
import ipdb
import random
import time
import tarfile
import datetime
import webdataset as wds
# import pyarrow.parquet as pq
import numpy as np
# import bbox_visualizer as bbv

from copy import deepcopy
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image
from webdataset import TarWriter
from joblib import Parallel, delayed
# from langdetect - detect
# from ftlangdetect import detect
# import cld3
from samplers import *
from tools import extract_frames, get_cache_video
from samplers import Un_sampler, KF_sampler

black_words = [
    'image unavailable',
    '.com', '.jpg', '.pdf', '.jpeg', 'png', 'tiff', 'svg',
]
tar_size = 500
# 设置环境变量
os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
process_dataset = ''

def load_image(image_path):
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
        
    return image



def process_tars(save_path, tar_name, samples,args=None):
    print(f"[{datetime.datetime.now()}] start to package {len(samples)} files to tar file {tar_name}")
    for tar_idx, tar_start in enumerate(tqdm(range(0, len(samples), tar_size))):
        tar_writer = TarWriter(os.path.join(save_path, f"{tar_name}-{tar_idx}.tar"))
        tar_samples = samples[tar_start: tar_start+tar_size]
        size = 0
        total = 0

        for file_idx, info in enumerate(tar_samples):
            # try:
            # ipdb.set_trace()
            import ipdb;ipdb.set_trace()
            image_dict_list = []
            image_name_list = []

            assert isinstance(info['video_path'], str)
            assert isinstance(info['value'], str)

            valid_count = 0
            
            # ==== Video Samples ====
            try:
                # TODO : 统一 一下均匀采样的return
                if args.type.lower() == 'un':
                    image_name_list, image_dict_list = Un_sampler(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'kf':
                    image_name_list, image_dict_list, indices_list, frame_types = KF_sampler(file_idx, info['video_path'],args=args)
                else:
                    raise ValueError(f"args.type {args.type} is not supported")

            except:
                print(f"Error when processing video {info['video_path']}")
                continue
    
            # ==== Conversation Info ====
            
            if args.type.lower() == 'un':
                human_value = "<image>"*len(image_name_list) 
            elif args.type.lower() == 'kf':
                human_value = ""
                for timestep, frame_type in zip(indices_list, frame_types):
                    human_value += f"<{frame_type}image>#{timestep}"
            else:
                raise ValueError(f"args.type {args.type} is not supported")
            
            human = {
                'from': 'human',
                "value": human_value,
            }
            
            gpt = {
                'from': 'gpt',
                "value": info['value'],
            }
            
            conversations_info = {
                "conversations":[human,gpt],
            }
            
            ipdb.set_trace()
            size += tar_writer.write(
                dict(
                    __key__=f"{file_idx:09d}",
                    json=dict(
                        prompt=human_value,
                        txt=info['value'],
                        info=info,
                        image_name_list=image_name_list
                    ),
            ))
            for image_dict in image_dict_list:
                size += tar_writer.write(image_dict)
            total += 1
                
            # except Exception as e:
            #     print(e)
            #     print(info)

        tar_writer.close()
        print(f"[{datetime.datetime.now()}] complete to write samples to tar file {tar_name}, size: {size}, nsamples: {total}")

def load_webvid():
    meta_data = json.load(open('/data/streamlit_source/raw_json/rmwm_webvid_QA_train_clean_train.json', 'r')) 
    print("Loaded webvid json")
    data = []
    for key in tqdm(range(len(meta_data['image'])),total=len(meta_data['image']),desc='Converting the Webvid format to required format...'):
        video_path = meta_data['image'][str(key)]
        caption = meta_data['value'][str(key)]
        data.append({
            'video_path': video_path,
            'value': caption
        })
    
    return data

def load_hd3m():
    meta_data = json.load(open('/data/streamlit_source/raw_json/path_to_output_hd-3m3.json', 'r')) 
    print("Loaded hd3m json")
    data = []
    for i in tqdm(meta_data, total=len(meta_data), desc='Converting hd3m format to required format...'):
        data.append({
            'video_path': i['video'],
            'value': i['caption']
        })
    return data

def load_internvid():
    import pandas as pd
    meta_data = pd.read_json('/data/streamlit_source/raw_json/InternVid-10M-FLT-INFO.jsonl', lines=True)
    print("Loaded internvid json")
    data = []
    
    for idx in tqdm(range(len(meta_data)), total=len(meta_data), desc='Converting internvid format to required format...'):
        file_name = os.path.join("_".join([meta_data['YoutubeID'][idx],meta_data['Start_timestamp'][idx],meta_data['End_timestamp'][idx]]) + ".mp4")
        caption = meta_data['Caption'][idx]
        
        data.append({
            'video_path': file_name,
            'value': caption
        })
    return data


def job(dataset, num_jobs=64, machine_id=0, total_machine=1,args=None):
    global process_dataset
    process_dataset = dataset
    
    if dataset == 'webvid':
        data = load_webvid()
        save_path = f"/mnt/shared-storage/tenant/hypertext/kanelin/webvid"
    elif dataset == 'hd3m':
        data = load_hd3m()
        save_path = f"/mnt/shared-storage/tenant/hypertext/kanelin/hd3m"
    elif dataset == 'internvid':
        data = load_internvid()
        save_path = f"/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/pack"
    else:
        raise ValueError(f"dataset {dataset} is not supported")

    print(f'{len(data)} samples in total')
    data = data[machine_id::total_machine]
    print(f'{len(data)} samples for machine {machine_id} with {total_machine} machines in total')
    
    if len(data) % (num_jobs) != 0:
        truncated_length = int(len(data) // num_jobs + 1) * num_jobs
    else:
        truncated_length = len(data)
    print(f"after truncating, {len(data)} files has been shrunk to {truncated_length}")

    start_time = time.time()
    # pattern = re.compile(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9^.^\-^+^*^/^$^,^，^。^!^！^?^？^:^：^;^；^\(^（^\)^）^【^】^《^》^…^ ]')
    # pattern = r"\b(?:https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+(?:[/?].*)?"
    
    Path(save_path).mkdir(parents=True, exist_ok=True)

    per_job_size = truncated_length // num_jobs
    print(f'per job size will be 1 and {num_jobs} works in parallel!')
    print(f'total job size will be {per_job_size} == {truncated_length} / ({num_jobs})')

    # for i in range(0, truncated_length, per_job_size):
    #     process_tars(
    #         save_path, 
    #         f"shard-{machine_id}-{i}-{i+per_job_size}",
    #         data[i:i+per_job_size], 
    #     )

    Parallel(n_jobs=num_jobs)(delayed(process_tars)(
        save_path, 
        f"shard-{machine_id}-{i}-{i+per_job_size}",
        data[i:i+per_job_size], 
    ) for i in range(0, truncated_length, per_job_size))

    end_time = time.time()
    print(f"The precessing procedure for {len(data)} files ran for {(end_time - start_time)} seconds")

def debug_online_data():
    # save_path = "/data/webvid/debug/tmp/"
    # tar_name = "test"
    # data = load_webvid()
    # data = load_hd3m()
    # random.shuffle(data)
    # samples = data[:1000]
    
    # wrong_case = "G11j0f3HkdE/G11j0f3HkdE.18_0.mp4"
    
    # process_tars(save_path, tar_name, samples)
    
    # ================ debug broken videos ==================
    # video_prefix = "s3://vision-language-data/video-data/hd130m/process_videos/"
    # broken_video = "Ob80WT1NhcY/Ob80WT1NhcY.42_2.mp4"
    # video_path = os.path.join(video_prefix, broken_video)
    # cache_path = get_cache_video(video_path)
    # images = extract_frames(cache_path)
    
    image_name_list, image_dict_list = Un_sampler(0, broken_video)
    import ipdb;ipdb.set_trace()

def debug_local_data(args):
    save_path = "/data/webvid/debug/tmp/"
    tar_name = "test-" + time.strftime("%Y%m%d-%H%M%S")
    data = load_internvid()
    samples = data[:1000]
    process_tars(save_path, tar_name, samples, args)
    
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_id", type=int, default=0)
    parser.add_argument("--total_machine", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="internvid",help="webvid, hd3m, internvid etc.")
    parser.add_argument("--workers", type=int, default=64) # 64
    parser.add_argument("--type", type=str, default="kf",help="un for Uniform sampling, kf for I&P sampling")
    parser.add_argument("--total_frames", type=int, default=24, help="The total number of frames to extract from a video")
    parser.add_argument("--Iframes", type=int, default=8, help="The number of keyframes to extract from a video") 
    parser.add_argument("--time_scale", type=int, default=1000, help="Scale of relative timestamps") 
    args = parser.parse_args()
    
    
    # job(dataset=args.dataset, num_jobs=args.workers, machine_id=args.machine_id, total_machine=args.total_machine,args=args)
    
    debug_local_data(args)