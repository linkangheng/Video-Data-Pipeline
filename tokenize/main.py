# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Megvii, Inc. All rights reserved.
# ------------------------------------------------------------------------------------------------
import os
import argparse
import datetime
import random
import ipdb
import json
import time
import megfile
import webdataset as wds

from tqdm import tqdm
from webdataset import TarWriter
from tokenizer import StepmmTokenizer, Llama2mmTokenizer
from joblib import Parallel, delayed
from pathlib import Path

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
KEYFRAME_IMAGE_TOKEN = ["<Iimage>","<Pimage>"]
DEFAULT_DREAM_TOKEN = "<dream>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_DREAM_START_TOKEN = "<dream_start>" # NOTE make llm dream!
DEFAULT_DREAM_END_TOKEN = "<dream_end>" # NOTE make llm dream!
special_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN

data_dict = {
    'howtointerlink-un-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/lucaszhao/howtointerlink-6m-tarfiles/',
    },
    'howtointerlink-kf-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/how2link/pack/kf',
    },
    'webvid-un-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/webvid/pack/un',
    },
    'webvid-kf-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/webvid/pack/kf',
    },
    'hd3m-un-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/hdvila/pack/un',
    },
    'hd3m-kf-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/hdvila/pack/kf',
    },
    'internvid-un-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/pack/un',
    },
    'internvid-kf-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/pack/kf',
    },
    'merlin-s-how2link-debug': {
        'path': '/data/video_pack/debug/data/5_27',
    },
    'merlin-s-how2link': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/data/how2link/interleave/anno/video_interleave/annotations/how2link/packing/pack',
    },
    'momentor-un-v1': {
        'path': '/mnt/shared-storage/tenant/hypertext/kanelin/hd3m/',
    },
    # 'momentor-kf-v1': {
    #     'path': '/mnt/shared-storage/tenant/hypertext/kanelin/hd3m/',
    # }
}

def save_to_tar(filename, stream, cached_data):
    cached_json = cached_data[0]
    for data in cached_data[1:]:
        cached_json["image"].extend(data['image'])
        cached_json["input_ids"].extend(data['input_ids'][1:])
        cached_json["loss_mask"].extend(data['loss_mask'][1:])
        cached_json["text"].extend(data['text'])

    images = cached_json.pop('image')
    cached_json['image'] = [f"{idx}" for idx in range(len(images))]
    stream.write({"__key__": filename, "json": cached_json})
    for idx, image in enumerate(images):
        stream.write({"__key__": f"{filename}-{idx}", "jpg": image})

def special_count(tokenized_value):
    # if sample_type == "kf":
    #     return tokenized_value.count('<Iimage>') + tokenized_value.count('<Pimage>')
    # elif sample_type == "un":
    #     return tokenized_value.count(special_token)
    return tokenized_value.count(special_token)

def add_image_token(text, sample_type):
    # deal with ambiguous image token in track (wo \n) and detection (w. \n) data
    # if DEFAULT_IMAGE_TOKEN + '\n' in text:
    #     text = text.replace(DEFAULT_IMAGE_TOKEN, special_token)
    # elif DEFAULT_IMAGE_TOKEN in text:
    #     text = text.replace(DEFAULT_IMAGE_TOKEN, special_token + '\n')
    # else:
    #     text = special_token + '\n' + text
    #  ---------------------------------------------------------------------------
    if sample_type == "kf":
        for token in KEYFRAME_IMAGE_TOKEN:
            text = text.replace(token, special_token)
    elif sample_type == "un":
        text = text.replace(DEFAULT_IMAGE_TOKEN, special_token)
    elif sample_type == "merlin-s":
        pass
    else:
        raise ValueError("sample_type should be specified!")
    return text

def conversate(prompt, text, sample_type):
    return [
        {
            'from': 'human',
            'value': add_image_token(prompt, sample_type),
        },
        {
            'from': 'gpt',
            'value': text
        }
    ]

def conversate_multi(conversations):
    for conversation in conversations:
        if conversation['from'] == 'human':
            conversation['value'] = add_image_token(conversation['value'], "un")
    return conversations
    
def tokenize_conversation(conversation, tokenizer):
    input_ids = [1]
    loss_masks = [0]
    text = DEFAULT_BOS_TOKEN
    for idx, conv in enumerate(conversation):
        # assert isinstance(conv['value'], str), conversation
        tmp = conv['value'] + (DEFAULT_EOS_TOKEN if conv['from'] == 'gpt' else '')
        input_id = tokenizer.tokenize(tmp)

        label = [0 if conv['from'] == 'human' else 1] * len(input_id)

        text += ('\n' if idx else '') + tmp
        input_ids.extend(input_id)
        loss_masks.extend(label)

    return input_ids, loss_masks, text

def tokenize_and_merge_tarfiles(save_path, shard_name, tar_name, samples, tokenizer, sample_type):
    print(f"[{datetime.datetime.now()}] start to re-orgnize {len(samples)} tarfiles to shard {shard_name} tar file {tar_name}")
    Path(os.path.join(save_path, shard_name)).mkdir(parents=True, exist_ok=True)
    tar_writer = TarWriter(os.path.join(save_path, shard_name, f"{tar_name}.tar"))
    cached_data = []
    cached_key = ""
    cached_token_len = 0
    merged_samples_cnt = 0
    valid_cnt = 0

    for tar_idx, tar_path in enumerate(tqdm(samples)):
        
        dataset = iter(wds.DataPipeline(
            wds.SimpleShardList(tar_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode("pilrgb"),
            wds.to_dict("jpg;png;jpeg", "txt;json", handler=wds.warn_and_continue),
        ))
        for file_idx, sample in enumerate(dataset):
            cached_key = f"{str(tar_idx)}-{str(file_idx)}"
            # make sure the images_list and special_token are matched
            if sample_type == 'merlin-s':
                special_nums = 0
                for dialog in sample['json']['conversations']:
                    if dialog['from'] == 'human':
                        special_nums += dialog['value'].count("<image>")
                if special_nums != len(sample['json']['image_name_list']):
                    continue
            else:
                if special_count(conversate(sample['json']['prompt'], sample['json']['txt'], sample_type)[0]['value']) != len(sample['json']['image_name_list']):
                    # print(sample)
                    continue
            if sample_type == 'merlin-s':
                input_ids, loss_masks, text = tokenize_conversation(conversate_multi(sample['json']['conversations']), tokenizer)
            else:
                input_ids, loss_masks, text = tokenize_conversation(conversate(sample['json']['prompt'], sample['json']['txt'],sample_type), tokenizer)
            
            if cached_token_len + len(input_ids) > 8000 and len(cached_data) > 0:
                save_to_tar(cached_key, tar_writer, cached_data)
                merged_samples_cnt += 1

                cached_data = []
                cached_token_len = 0

            if len(input_ids) < 8000:
                image_list = []
                for image_name in sample['json']['image_name_list']:
                    img = sample[image_name.split('-')[-1] + '.jpg']
                    if max(img.size) > 1280:
                        # print(img.size)
                        img = img.resize((1280, 1280))
                    image_list.append(img)
                
                cached_data.append(dict(
                    image=image_list,
                    input_ids=input_ids,
                    loss_mask=loss_masks,
                    text=[text]
                ))
                cached_token_len += len(input_ids)

            valid_cnt += 1

    # ignore for avoiding duplicated key
    # if len(cached_data) > 0:
    #     save_to_tar(cached_key, tar_writer, cached_data)
    #     merged_samples_cnt += 1

    tar_writer.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {shard_name} tar file {tar_name}, nsamples: {valid_cnt}, merged nsamples: {merged_samples_cnt}")

def job(dataset_path, save_path, sample_type, num_jobs=64, start=0, end=1, shard_size=3):

    assert len(sample_type) > 0, "sample_type should be specified!"
    start_time = time.time()
    all_files = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if x.endswith('.tar')]
    print(len(all_files))
    all_files = all_files[start:end]
    print(f"from {start} to {end}, totally {len(all_files)} files to be processed")
    
    if len(all_files) % (num_jobs * shard_size) != 0:
        truncated_length = int(len(all_files) // num_jobs // shard_size + 1) * num_jobs * shard_size
    else:
        truncated_length = len(all_files)
    print(f"after truncating, {len(all_files)} files has been shrunk to {truncated_length}")

    per_job_size = truncated_length // num_jobs // shard_size
    print(f'per job size will be {shard_size} and {num_jobs} workers work in parallel!')
    print(f'total job execetions will be {per_job_size} == {truncated_length} / ({num_jobs} * {shard_size})')

    tokenizer_model = "/mnt/shared-storage/tenant/open-source/Llama-2-7b-hf/multimodal_tokenizer.model"
    tokenizer = Llama2mmTokenizer(tokenizer_model)
    
    for idx, x in enumerate(range(0, truncated_length, num_jobs * shard_size)):
        per_job_files = all_files[x: x+num_jobs * shard_size]
        
        # ========== for dubeg ==========
        # for i in range(0, len(per_job_files), shard_size):
        #     tokenize_and_merge_tarfiles(
        #         save_path,
        #         f"{start}-{end}", 
        #         f"shard_{idx}-{i}-{i+shard_size}",
        #         per_job_files[i:i+shard_size],
        #         tokenizer,
        #         sample_type
        #     )

        # ========== for multi-porcess ==========
        Parallel(n_jobs=num_jobs)(delayed(tokenize_and_merge_tarfiles)(
            save_path,
            f"{start}-{end}",
            f"shard_{idx}-{i}-{i+shard_size}",
            per_job_files[i:i+shard_size], 
            tokenizer,
            sample_type
        ) for i in range(0, len(per_job_files), shard_size))

    end_time = time.time()
    print(f"The precessing procedure for {len(all_files)} files ran for {(end_time - start_time)} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument('--dataset_name', type=str, default="howtointerlink-kf-v1")
    parser.add_argument('--sample_type', type=str, default="un", help="un, kf")
    args = parser.parse_args()
    
    job(
        dataset_path=data_dict[args.dataset_name]['path'],
        save_path=f'/mnt/shared-storage/tenant/hypertext/kanelin/tokenized_data/{args.dataset_name}-tokens_imgpatch_400_tokenlen_8k/',
        # save_path = "/mnt/shared-storage/tenant/hypertext/kanelin/data/how2link/interleave/anno/video_interleave/annotations/how2link/tokenized",
        num_jobs=args.workers,
        shard_size=args.shard_size, 
        start=args.start,
        end=args.end,
        sample_type=args.sample_type
    )