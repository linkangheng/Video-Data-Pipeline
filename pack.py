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
# from langdetect import detect
# from ftlangdetect import detect
# import cld3


black_words = [
    'image unavailable',
    '.com', '.jpg', '.pdf', '.jpeg', 'png', 'tiff', 'svg',
]
tar_size = 500


def load_image(image_path):
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
        
    return image

def video_sample(file_idx, video_path):
    # TODO: 根据给定的视频路径，返回视频的关键帧的PIL列表
    image_name_list = []
    
    return image_name_list

def process_tars(save_path, tar_name, samples):
    print(f"[{datetime.datetime.now()}] start to package {len(samples)} files to tar file {tar_name}")
    for tar_idx, tar_start in enumerate(tqdm(range(0, len(samples), tar_size))):
        tar_writer = TarWriter(os.path.join(save_path, f"{tar_name}-{tar_idx}.tar"))
        tar_samples = samples[tar_start: tar_start+tar_size]
        size = 0
        total = 0

        for file_idx, info in enumerate(tar_samples):
            # try:
            # ipdb.set_trace()
            image_dict_list = []
            image_name_list = []

            if isinstance(info['image'], str):
                info['image'] = [info['image']]

            assert isinstance(info['image'], list)

            valid_count = 0
            
            # ==== TODO: 构建info ====
            # ==== End ====
            
            # ==== TODO: 将这一部分的逻辑改为根据视频路径返回image_dict_list, image_name_list的一个函数 ====
            for index, img_path in enumerate(info['image']):
                try:
                    img = load_image(img_path.replace('kanelin/interlink7m/samples', 'vision-language-data/interlink7m'))
                    img.verify()
                    valid_count += 1
                except Exception as e:
                    print(e)
                    print(img_path)
                    continue

                image_name_list.append(f"{file_idx:09d}-{index}")
                image_dict_list.append(
                    dict(
                        __key__=f"{file_idx:09d}-{index}",
                        jpg=img,
                    )
                )
            # ==== End ====
            
            if valid_count != info['conversations'][0]['value'].count('<image>'):
                print('skip sample: ' + str(info))
                continue

            assert len(info['conversations']) == 2

            # ipdb.set_trace()
            size += tar_writer.write(
                dict(
                    __key__=f"{file_idx:09d}",
                    json=dict(
                        prompt=info['conversations'][0]['value'],
                        txt=info['conversations'][1]['value'],
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


def job(num_jobs=64, machine_id=0, total_machine=1):
    meta_data = json.load(open('/data/webvid/debug/data/rmwm_webvid_QA_train_clean_train.json', 'r')) 
    data = []
    
    print(f'Converting meta dict file to required format...')
    for key in tqdm(range(len(meta_data['image'])),total=len(meta_data['image'])):
        video_path = meta_data['image'][str(key)]
        caption = meta_data['value'][str(key)]
        data.append({
            'video_path': video_path,
            'value': caption
        })
    
    # random.shuffle(data)

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
    save_path = f"/data/webvid/debug/"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_id", type=int, default=0)
    parser.add_argument("--total_machine", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1) # 64
    args = parser.parse_args()

    job(num_jobs=args.workers, machine_id=args.machine_id, total_machine=args.total_machine)