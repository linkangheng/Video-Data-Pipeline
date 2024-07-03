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
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool

import webdataset as wds

from samplers import *
from tools import extract_frames, get_cache_video, merlin_s_qa_process
from dataset_loader import *
import time

# Constants
tar_size = 500

black_words = [
    'image unavailable',
    '.com', '.jpg', '.pdf', '.jpeg', 'png', 'tiff', 'svg',
]

video_data_types = ["video_text_pair", "video_interleave", "video_sft"]

def get_conversation(human_value, assistant_value):
    return [
        {
            'from': 'human',
            'value': human_value
        },
        {
            'from': 'assistant',
            'value': assistant_value
        }
    ]

def process_tars(save_path, tar_name, samples, args=None):
    print(f"[{datetime.datetime.now()}] start to package {len(samples)} files to tar file {tar_name}")
    for tar_idx, tar_start in enumerate(tqdm(range(0, len(samples), tar_size))):
        tar_writer = wds.TarWriter(os.path.join(save_path, f"{tar_name}-{tar_idx}.tar"))
        tar_samples = samples[tar_start: tar_start+tar_size]
        size = 0
        total = 0

        for file_idx, info in enumerate(tar_samples):
            # ==== Video Process ====
            try:
                if args.type.lower() == 'un':
                    image_name_list, image_dict_list = uniformSampler(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'kf':
                    image_name_list, image_dict_list, indices_list, frame_types = keyFrameSampler(file_idx, info['video_path'],args=args)
                elif args.type.lower() in video_data_types:
                    # 为命名统一，这里继续使用image_name_list和image_dict_list存放视频
                    image_name_list, image_dict_list = getVideoList(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'merlin-s':
                    image_name_list, image_dict_list = Merlin_S_sampler(file_idx, info['video_path'],args=args)
                else:
                    raise ValueError(f"sample types {args.type} is not supported")
            
            except Exception as e:
                import ipdb;ipdb.set_trace()
                print(f"Error when processing video {info['video_path']}")
                continue
    
            # ==== Conversation Process ====
            if args.type.lower() == 'un':
                # here for uniform sampling
                human_value = "<image>"*len(image_name_list) 

            elif args.type.lower() == 'kf':
                # here for keyframe sampling
                human_value = ""
                for timestep, frame_type in zip(indices_list, frame_types):
                    human_value += f"<{frame_type}image>#{timestep}"

            elif args.type.lower() == 'video_text_pair':
                # here for video-text pair
                human_value = "<video>"
                conversations = get_conversation(human_value, info['value'])

            elif args.type.lower() == 'video_interleave':
                #TODO: make conversations
                pass
            
            elif args.type.lower() == 'video_sft':
                #TODO: make conversations
                pass
                
            elif args.type.lower() == 'merlin-s':
                # here for merlin-s dataset
                try:
                    questions, answers = merlin_s_qa_process(info['value'])
                except:
                    print(info)
                    continue
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

            elif args.type.lower() in video_data_types:
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
                            conversations = conversations,
                            videos = image_name_list
                        ),
                ))
                pass

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
    if dataset == 'webvid':
        data = load_webvid()
    elif dataset == 'hd3m':
        data = load_hd3m()
    elif dataset == 'internvid':
        data = load_internvid()
    elif dataset == 'how2link':
        data = load_how2link()
    elif dataset == 'ego4d':
        data = load_ego4d()
    elif dataset == 'merlin-s':
        data = load_merlin(args.interleave_path)
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
    # for i in range(0, truncated_length, per_job_size):
    #     process_tars(
    #         save_path, 
    #         f"shard-{machine_id}-{i}-{i+per_job_size}",
    #         data[i:i+per_job_size], 
    #         args=args
    #     )

    #  ============== 多线程运行 ==============
    Parallel(n_jobs=num_jobs)(delayed(process_tars)(
        save_path, 
        f"shard-{machine_id}-{i}-5-{i+per_job_size}",
        data[i:i+per_job_size], 
        args=args
    ) for i in range(0, truncated_length, per_job_size))
    
    #  ============== 支持ffmpeg ==============
    # with Pool(num_jobs) as pool:
    #     results = pool.starmap(process_tars, [
    #         (save_path, f"shard-{machine_id}-{i}-{i+per_job_size}", data[i:i+per_job_size], args)
    #         for i in range(0, truncated_length, per_job_size)
    #     ])
    # pool.close()
    # pool.join()

    end_time = time.time()
    print(f"The precessing procedure for {len(data)} files ran for {(end_time - start_time)} seconds")
    
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


    

