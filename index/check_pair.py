# ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Megvii, Inc. All rights reserved.
# ------------------------------------------------------------------------------------------------
import os
import argparse
import megfile
import pickle
import random
import ipdb
import webdataset as wds

from tokenizer import Llama2mmTokenizer
from joblib import Parallel, delayed
from pathlib import Path

def dump_sample_for_checking(dataset_name, file_path, sample, tokenizer):
    tar_name = file_path.split('/')[-1].split('.')[0]
    Path(os.path.join('archive', dataset_name, tar_name, 'images')).mkdir(parents=True, exist_ok=True)
    
    open(os.path.join('archive', dataset_name, tar_name, 'ground_truth.txt'), 'w').writelines('\n'.join(sample['json']['text']))
    open(os.path.join('archive', dataset_name, tar_name, 'detokenized_tokens.txt'), 'w').writelines(tokenizer.detokenize(sample['json']['input_ids']))
    open(os.path.join('archive', dataset_name, tar_name, 'unmasked_tokens.txt'), 'w').writelines(
        tokenizer.detokenize([token for token, mask in zip(sample['json']['input_ids'], sample['json']['loss_mask']) if mask == 1])
    )
    for image_name in sample['json']['image']:
        sample[f'{image_name}.jpg'].save(os.path.join('archive', dataset_name, tar_name, 'images', f'{image_name}.jpg'))

def index_tar(dataset_name, file_path, tokenizer):
    dataset = iter(wds.DataPipeline(
        wds.SimpleShardList(file_path),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode("pilrgb"),
        wds.to_dict("jpg;png;jpeg", "txt;json", handler=wds.warn_and_continue),
    ))
    valid_size = 0
    for sample in dataset:
        assert len(sample['json']['input_ids']) == len(sample['json']['loss_mask']) and len(sample['json']['input_ids']) < 8000
        assert sample['json']['input_ids'].count(32001) == sample['json']['input_ids'].count(32002)
        # assert sample['json']['input_ids'].count(17) == sample['json']['input_ids'].count(18)
        assert sample['json']['input_ids'].count(32000) == len(sample['json']['image']) * (256)
        if valid_size == 0 and random.random() < 0.01:
            dump_sample_for_checking(dataset_name, file_path, sample, tokenizer)

        valid_size += 1
    # print(valid_size)
    return dict(url=file_path, nsamples=valid_size)

def job(dataset_name, dataset_path, num_jobs=64):
    all_files = []
    for dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, dir)):
            all_files += [os.path.join(dataset_path, dir, x) for x in os.listdir(os.path.join(dataset_path, dir)) if x.endswith('.tar')]
    
    print(f"totally {len(all_files)} files to be processed")
    
    tokenizer_model = "/mnt/shared-storage/tenant/open-source/Llama-2-7b-hf/multimodal_tokenizer.model"
    tokenizer = Llama2mmTokenizer(tokenizer_model)

    outputs = []
    results = Parallel(n_jobs=num_jobs)(delayed(index_tar)(dataset_name, tar_path, tokenizer) for tar_path in all_files)
    for _, res in enumerate(results):
        outputs.append(res)

    pickle.dump(outputs, open(f"/mnt/shared-storage/tenant/hypertext/kanelin/index_data/index-v1-llama/{dataset_name}.pkl", "wb"))
    print(f"{sum([x['nsamples'] for x in outputs])} pairs ({dataset_name}) in total for with {len(outputs)} tarfiles")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument('--dataset_name', type=str, default="xiachufang")
    args = parser.parse_args()

    job(
        dataset_name=args.dataset_name,
        dataset_path=f'/mnt/shared-storage/tenant/hypertext/kanelin/tokenized_data/{args.dataset_name}-tokens_imgpatch_400_tokenlen_8k/',
        num_jobs=args.workers, 
    )
    
    # tokenizer_model = "/mnt/shared-storage/tenant/open-source/Llama-2-7b-hf/multimodal_tokenizer.model"
    # tokenizer = Llama2mmTokenizer(tokenizer_model)
    # print('[INFO] Start indexing...')
    # res = index_tar('webvid-kf-v1', '/mnt/shared-storage/tenant/hypertext/kanelin/tokenized_data/webvid-kf-v1-tokens_imgpatch_400_tokenlen_8k/6000-7000/shard_1-50-55.tar', tokenizer)
    # print(res)
    
    
    
    
    