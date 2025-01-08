# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Megvii, Inc. All rights reserved.
#  conda activate webvid
#  python pack.py --dataset unicontrol --workers 64 --type unicontrol --save_path /mnt/jfs-test/data/unicontrol/tars/aesthetics_plus_all_group_bbox_all --total_machine 16 --machine_id 0
#  python pack.py --dataset internvid --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/pack/kf --total_machine 16 --machine_id  
#  python pack.py --dataset webvid --workers 64 --type video-only --save_path /data/video_pack/debug/data/7_10  --machine_id 0 --total_machine 8
#  python pack.py --dataset hd3m --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/hdvila/pack/kf --machine_id 
#  python pack.py --dataset internvid --workers 64 --type un --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/un --machine_id 
#  python pack.py --dataset how2link --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/how2link/kf
#  python pack.py --dataset ego4d --workers 64 --type kf --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/ego4d/pack/kf --total_machine 8 --machine_id 0
#  python pack.py --machine_id 0 --total_machine 8 --dataset  videochat2 --workers 64 --type image --save_path /mnt/shared-storage/tenant/hypertext/danielyu/data/VideoChat2/packed_tar
#  cd /data/video_pack/pack && conda activate webvid && python pack.py --dataset videochat2 --workers 64 --type merlin-un --save_path /mnt/shared-storage/tenant/hypertext/kanelin/pack/videochat2_full  --total_machine 8 --machine_id 
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



def deafult_conversation(human_value, assistant_value):
    return [
        {
            'from': 'human',
            'value': human_value
        },
        {
            'from': 'gpt',
            'value': assistant_value
        }
    ]

def get_question(qa):
    question = ""
    for key, value in qa.items():
        if key != 'a':
            question += value
    return question

def videochat2_conversation(qa_li, type = "image", num_samples = -1):
    # support image/video
    special_token = f"<image>" if type == "image" else f"<video>"*num_samples
    conversations = []
    for idx, qa in enumerate(qa_li):
        human_value = get_question(qa)
        assistant_value = qa['a']
        if idx == 0:
            human_value += special_token
        conversations.extend(deafult_conversation(human_value, assistant_value))
    return conversations

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
                elif args.type.lower() == 'image':
                    image_name_list, image_dict_list = get_image(file_idx, info['video_path'],args=args)
                elif args.type.lower() == 'merlin-un':
                    image_name_list, image_dict_list = get_images(file_idx, info['images'])
                elif args.type.lower() == 'unicontrol':
                    image_name_list, image_dict_list = get_unicontrol_images(file_idx, info['source'], info['target'])
                else:
                    raise ValueError(f"sample types {args.type} is not supported")
            except Exception as e:
                print(f"Error when processing video {info}") 
                continue
    
            # ==== Conversation Process ====
            if args.type.lower() == 'un':
                # here for uniform sampling
                human_value = "<image>"*len(image_name_list) 
                conversations = deafult_conversation(human_value, info['value'])

            elif args.type.lower() == 'kf':
                # here for keyframe sampling
                human_value = ""
                for timestep, frame_type in zip(indices_list, frame_types):
                    human_value += f"<{frame_type}image>#{timestep}"

            elif args.type.lower() == 'video_text_pair':
                # here for video-text pair
                human_value = "<video>"
                conversations = deafult_conversation(human_value, info['value'])
            
            elif args.type.lower() == 'image':
                conversations = videochat2_conversation(info['value'])

            elif args.type.lower() == 'video_interleave':
                #TODO: make conversations
                pass
            
            elif args.type.lower() == 'video_sft':
                #TODO: make conversations
                pass
            
            elif args.type.lower() == 'merlin-un':
                # TODO: prepare the prompt and txt for merlin-un dataset
                conversations = info['conversations']
                prompt = []
                txt = []
                for idx in range(len(conversations)):
                    if idx % 2 == 0 and conversations[idx]['from'] == 'human':
                        prompt.append(conversations[idx]['value'])
                    elif idx % 2 != 0 and conversations[idx]['from'] == 'gpt':
                        txt.append(conversations[idx]['value'])

            elif args.type.lower() == 'merlin-s':
                # here for merlin-s dataset
                try:
                    questions, answers = merlin_s_qa_process(info['value'])
                except:
                    print(info)
                    continue
                conversations = []
                info = {}
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

            elif args.type.lower() == 'image':
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
                            conversations = conversations,
                            image = image_name_list
                        ),
                ))
            
            elif args.type.lower() == 'unicontrol':
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
                            source = image_name_list[0],
                            target = image_name_list[1],
                            prompt = info['value']
                        ),
                ))

            elif args.type.lower() == 'merlin-un':
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
                            prompt=prompt,
                            txt=txt,
                            info=info,
                            image_name_list=image_name_list
                        ),
                ))
            else:
                size += tar_writer.write(
                    dict(
                        __key__=f"{file_idx:09d}",
                        json=dict(
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
    
    # prepare for meta dataset
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
    elif dataset == 'llava_pretrain':
        data = load_llava()
    elif dataset == 'merlin-s':
        data = load_merlin(args.interleave_path)
    elif dataset.lower() == 'videochat2':
        data = load_videochat2()
    elif dataset.lower() == 'unicontrol':
        data = load_unicontrol(subset=args.subset)
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
        f"shard-{machine_id}-{i}-{i+per_job_size}",
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
    parser.add_argument("--dataset", type=str, default="internvid",help="webvid, hd3m, internvid, how2link, ego4d, llava_pretrain, unicontrol etc.")
    parser.add_argument("--workers", type=int, default=64) # 64
    parser.add_argument("--type", type=str, default="video-only",help="un, kf, video-only, image, unicontrol; un for Uniform sampling, kf for I&P sampling")
    parser.add_argument("--total_frames", type=int, default=24, help="The total number of frames to extract from a video")
    parser.add_argument("--Iframes", type=int, default=8, help="The number of keyframes to extract from a video")
    parser.add_argument("--time_scale", type=int, default=1000, help="Scale of relative timestamps")
    parser.add_argument("--interleave_path", type=str, default="", help="The path of interleave json file")
    parser.add_argument("--save_path", type=str, default="/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/un", help="Path to save the tar files") 
    parser.add_argument("--subset", type=str, default="aesthetics_plus_all_group_bbox_all", help="The subset of the dataset")
    args = parser.parse_args()
    
    job(dataset=args.dataset, num_jobs=args.workers, machine_id=args.machine_id, total_machine=args.total_machine,args=args)
