# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Megvii, Inc. All rights reserved.
#  conda activate webvid
#  python /data/webvid/pack.py --dataset internvid --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/pack/kf --total_machine 16 --machine_id  
#  python /data/webvid/pack.py --dataset webvid --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/webvid/pack/kf  --machine_id 
#  python /data/webvid/pack.py --dataset hd3m --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/hdvila/pack/kf --machine_id 
#  python /data/webvid/pack.py --dataset internvid --workers 64 --type un --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/un --machine_id 
#  sudo python /data/webvid/pack.py --dataset how2link --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/how2link/kf
#  python /data/webvid/pack.py --dataset ego4d --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/ego4d/pack/kf --total_machine 8 --machine_id 0
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
from multiprocessing import Pool, cpu_count

# import bbox_visualizer as bbv

from copy import deepcopy
import ijson
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
from tools import extract_frames, get_cache_video, merlin_s_qa_process
from samplers import Un_sampler, KF_sampler, Merlin_S_sampler
import threading


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

def load_webvid():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
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

def load_ego4d():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    meta_data = json.load(open('/data/streamlit_source/raw_json/ego4d.json', 'r')) 
    print("Loaded ego3d json")
    data = []
    for i in tqdm(meta_data, total=len(meta_data), desc='Converting ego3d format to required format...'):
        data.append({
            'video_path': i['video_path'],
            'value': i['caption']
        })
    
    return data

def load_hd3m():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    meta_data = json.load(open('/data/streamlit_source/raw_json/path_to_output_hd-3m3.json', 'r')) 
    print("Loaded hd3m json")
    data = []
    for i in tqdm(meta_data, total=len(meta_data), desc='Converting hd3m format to required format...'):
        data.append({
            'video_path': i['video'],
            'value': i['caption']
        })
    return data

def load_how2link():
    os.environ['OSS_ENDPOINT'] = 'http://tos-s3-cn-shanghai.ivolces.com'
    json_path = "/data/streamlit_source/raw_json/How2link.json"
    data = []
    
    with open(json_path, 'r') as f:
        for record in tqdm(ijson.items(f, 'item'), desc='Converting the How2link format to required format...',total=932157):
            for clip in record['clips']:
                clip_path = "/".join(clip['clip_path'].split("/")[-3:]) + ".mp4"
                caption = clip['caption']
                data.append({
                    'video_path': clip_path,
                    'value': caption
                })
                
    return data

def load_internvid():
    import pandas as pd
    #  debug: /data/webvid/debug/data/InternVid-10M-FLT-INFO-top10.jsonl
    #  real: /data/streamlit_source/raw_json/InternVid-10M-FLT-INFO.jsonl
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
    # import ipdb;ipdb.set_trace()
    return data

def load_sft(sft_path):
    meta_data = json.load(open(sft_path))
    data = []
    for i in tqdm(meta_data, total=len(meta_data), desc='Converting hd3m format to required format...'):
        data.append({
            'video_path': i['video'],
            'value': i['QA']
        })
    return data

def load_merlin(interleave_path):
    # This function is for merlin-s dataset
    if not os.path.exists(interleave_path):
        raise ValueError(f"interleave file {interleave_path} does not exist")
    data = []
    with open(interleave_path, 'r') as f:
        for record in tqdm(ijson.items(f, "item"), desc="processing the merlin-s dataset..."):
            data.append({
                'video_path': record['image_info'],
                'value': record['text_list']
            })
    return data

def process_tars(save_path, tar_name, samples, args=None):
    print(f"[{datetime.datetime.now()}] start to package {len(samples)} files to tar file {tar_name}")
    for tar_idx, tar_start in enumerate(tqdm(range(0, len(samples), tar_size))):
        tar_writer = TarWriter(os.path.join(save_path, f"{tar_name}-{tar_idx}.tar"))
        tar_samples = samples[tar_start: tar_start+tar_size]
        size = 0
        total = 0

        for file_idx, info in enumerate(tar_samples):
            # ipdb.set_trace()
            # assert isinstance(info['video_path'], str)
            # assert isinstance(info['value'], str)

            valid_count = 0
            
            # ==== Video Samples ====
            try:
                # TODO : 支持视频的打包
                if args.type.lower() == 'un':
                    image_name_list, image_dict_list = Un_sampler(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'kf':
                    image_name_list, image_dict_list, indices_list, frame_types = KF_sampler(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'video-only':
                    image_name_list, image_dict_list = Video_Reader(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'merlin-s':
                    image_name_list, image_dict_list = Merlin_S_sampler(file_idx, info['video_path'],args=args)
                else:
                    raise ValueError(f"sample types {args.type} is not supported")
                # print(f"Successfully processed video samples {info['video_path']}")
            except Exception as e:
                import ipdb;ipdb.set_trace()
                print(e)
                print(f"Error when processing video {info['video_path']}")
                continue
    
            # ==== Conversation Info ====
            
            if args.type.lower() == 'un':
                # here for uniform sampling
                human_value = "<image>"*len(image_name_list) 

            elif args.type.lower() == 'kf':
                # here for keyframe sampling
                human_value = ""
                for timestep, frame_type in zip(indices_list, frame_types):
                    human_value += f"<{frame_type}image>#{timestep}"

            elif args.type.lower() == 'video-only':
                # here for video-only packing, which is for cherry Han
                human_value = ""

            elif args.type.lower() == 'merlin-s':
                # here for merlin-s dataset
                questions, answers = merlin_s_qa_process(info['value'])
                conversations = []
                for question, answer in zip(questions, answers):
                    # Human conversation
                    conversations.append({
                        'from': 'human',
                        'value': question,
                    })
                    # GTP conversation
                    conversations.append({
                        'from': 'gpt',
                        'value': answer,
                    })
            
            else:
                raise ValueError(f"args.type {args.type} is not supported")
            
            # ==== Packing the samples ====
            if args.type.lower() == 'video-only':
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
                            caption=info['value'],
                            video_id=image_name_list[0]
                        ),
                ))

            elif args.type.lower() == 'merlin-s':
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
                            conversations = conversations,
                            image_name_list = image_name_list
                        ),
                ))

            else:
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
                
        tar_writer.close()
        print(f"[{datetime.datetime.now()}] complete to write samples to tar file {tar_name}, size: {size}, nsamples: {total}")


def job(dataset, num_jobs=64, machine_id=0, total_machine=1,args=None):
    print(args)
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
    elif dataset == 'how2link':
        data = load_how2link()
        save_path = f"/mnt/shared-storage/tenant/hypertext/kanelin/data/how2link/pack/"
    elif dataset == 'ego4d':
        data = load_ego4d()
        save_path = f"/mnt/shared-storage/tenant/hypertext/kanelin/data/ego4d/pack/kf"
    elif dataset == 'merlin-s':
        data = load_merlin(args.interleave_path)
        save_path = f"/mnt/shared-storage/tenant/hypertext/kanelin/data/merlin-s/pack"
    else:
        try:
            data = load_sft(dataset)
        except:
            raise ValueError(f"dataset {dataset} is not supported")

    if args.save_path:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
        save_path = args.save_path

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
    print(f'per job size will be {per_job_size} and {num_jobs} works in parallel!')
    print(f'total job size will be {per_job_size} == {truncated_length} / ({num_jobs})')
    
    # ============== 单线程调试 ============== 
    for i in range(0, truncated_length, per_job_size):
        process_tars(
            save_path, 
            f"shard-{machine_id}-{i}-{i+per_job_size}",
            data[i:i+per_job_size], 
            args=args
        )

    #  ============== 只支持均匀抽帧 ==============
    # Parallel(n_jobs=num_jobs)(delayed(process_tars)(
    #     save_path, 
    #     f"shard-{machine_id}-{i}-{i+per_job_size}",
    #     data[i:i+per_job_size], 
    #     args=args
    # ) for i in range(0, truncated_length, per_job_size))
    
    #  ============== 支持ffmpeg ==============
    # with Pool(num_jobs) as pool:
    #     # 创建一个迭代器，用于生成每个进程的任务参数
    #     # 这里使用星号(*)来解包args，使其作为独立的参数传递给process_tars函数
    #     results = pool.starmap(process_tars, [
    #         (save_path, f"shard-{machine_id}-{i}-{i+per_job_size}", data[i:i+per_job_size], args)
    #         for i in range(0, truncated_length, per_job_size)
    #     ])
    # pool.close()
    # pool.join()

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
    tar_name = "/data/video_pack/debug/tmp/test-20240509-173018-0.tar"
    # tar_name = "test-" + time.strftime("%Y%m%d-%H%M%S")
    data = load_internvid()
    samples = data[:1000]
    process_tars(save_path, tar_name, samples, args)
    
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_id", type=int, default=0)
    parser.add_argument("--total_machine", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="internvid",help="webvid, hd3m, internvid, how2link, ego4d etc.")
    parser.add_argument("--workers", type=int, default=64) # 64
    parser.add_argument("--type", type=str, default="video-only",help="un, kf, video-only; un for Uniform sampling, kf for I&P sampling")
    parser.add_argument("--total_frames", type=int, default=24, help="The total number of frames to extract from a video")
    parser.add_argument("--Iframes", type=int, default=8, help="The number of keyframes to extract from a video")
    parser.add_argument("--time_scale", type=int, default=1000, help="Scale of relative timestamps")
    parser.add_argument("--interleave_path", type=str, default="", help="The path of interleave json file")
    parser.add_argument("--save_path", type=str, default="/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/un", help="Path to save the tar files") 
    args = parser.parse_args()
    
    job(dataset=args.dataset, num_jobs=args.workers, machine_id=args.machine_id, total_machine=args.total_machine,args=args)


    

