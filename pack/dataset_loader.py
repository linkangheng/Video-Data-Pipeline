import os
import json
import pandas as pd
from tqdm import tqdm
import json
from PIL import Image
from io import BytesIO
import megfile
import glob
from datasets import load_dataset,Features,Value,Image

data_prefix = {
    "webvid": "s3://vision-language-data/video-data/webvid10m/process_videos/",
    "hd3m": "s3://vision-language-data/video-data/hd130m/process_videos/",
    "internvid": "/mnt/shared-storage/tenant/hypertext/kanelin/data/internvid/raw/raw/InternVId-FLT",
    "how2link": "s3://kanelin/interlink7m/",
    "llava_pretrain": "/mnt/shared-storage/tenant/hypertext/danielyu/data/LLaVA-Pretrain/image",
    "ego4d": "",
    "momentor": "s3://kanelin/video_data/Momentor/video"
}

def get_prefix(dataset):
    if dataset in data_prefix.keys():
        return data_prefix[dataset]
    else:
        return ""

def load_image(image_path):
    if 's3://' in image_path:
        with megfile.smart_open(image_path, "rb") as f:
            bytes_data = f.read()
        image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

def load_hf_dataset(dataset_path,split="train"):
    try:
        if os.path.isdir(dataset_path):
            data_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                        if f.endswith('.parquet')]
        else:
            data_files = [dataset_path]
        cache_dir = f"/mnt/jfs/hypertext/nextstep-project/image_datasets/inbox/image_edit/huggingface_cache"
        # 加载数据集
        dataset = load_dataset("parquet", data_files=data_files,split=split,num_proc=64,cache_dir=cache_dir,columns=["omni_edit_id","src_img","edited_img","edited_prompt_list"])
        return dataset
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return None

def load_webvid():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    meta_data = json.load(open('/data/streamlit_source/raw_json/rmwm_webvid_QA_train_clean_train.json', 'r')) 
    print("Loaded webvid json")
    data = []
    for key in tqdm(range(len(meta_data['image'])),total=len(meta_data['image']),desc='Converting the Webvid format to required format...'):
        video_path = os.path.join(get_prefix('webvid'), meta_data['image'][str(key)])
        caption = meta_data['value'][str(key)]
        data.append({
            'video_path': video_path,
            'value': caption
        })
    return data

def load_cambrain():
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    meta_data = json.load(open('/mnt/shared-storage/tenant/hypertext/danielyu/data/Cambrian-10M/jsons/Cambrian7M_withsystemprompt.json', 'r')) 
    print("Loaded webvid json")
    data = []
    for key in tqdm(range(len(meta_data['image'])),total=len(meta_data['image']),desc='Converting the Webvid format to required format...'):
        video_path = os.path.join(get_prefix('webvid'), meta_data['image'][str(key)])
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

def load_unicontrol(subset):
    json_dir = "/mnt/jfs-test/data/unicontrol/json_files"
    source_prefix = "/mnt/jfs-test/data/unicontrol/images"
    target_prefix = "/mnt/jfs-test/data/unicontrol/conditions"
    
    def get_target_key(data):
        for k,v in data.items():
            if "control_" in k:
                return k
        return None
    
    jsonl_file = os.path.join(json_dir, f"{subset}.json")
    datas = []
    with open(jsonl_file, 'r') as f:
        for line in tqdm(f, desc='Converting unicontrol format to required format...'):
            data = json.loads(line)
            source_image = os.path.join(source_prefix, data['source'])
            target_key = get_target_key(data)
            target_image = os.path.join(target_prefix, data[target_key])
            datas.append({
                'source': source_image,
                'target': target_image,
                'value': data['prompt']
            })
    return datas

def load_hd3m():
    os.environ['AWS_PROFILE'] = 'default'
    os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
    meta_data = json.load(open('/data/streamlit_source/raw_json/path_to_output_hd-3m3.json', 'r')) 
    print("Loaded hd3m json")
    data = []
    for i in tqdm(meta_data, total=len(meta_data), desc='Converting hd3m format to required format...'):
        data.append({
            'video_path': os.path.join(get_prefix('hd3m'), i['video']),
            'value': i['caption']
        })
    return data

def load_how2link():
    os.environ['AWS_PROFILE'] = 'tos'
    os.environ['OSS_ENDPOINT'] = 'http://tos-s3-cn-shanghai.ivolces.com'
    json_path = "/data/streamlit_source/raw_json/How2link.json"
    data = []
    
    with open(json_path, 'r') as f:
        for record in tqdm(ijson.items(f, 'item'), desc='Converting the How2link format to required format...',total=932157):
            for clip in record['clips']:
                clip_path = "/".join(clip['clip_path'].split("/")[-3:]) + ".mp4"
                caption = clip['caption']
                data.append({
                    'video_path': os.path.join(get_prefix('how2link'), clip_path),
                    'value': caption
                })
                
    return data

def load_internvid():
    import pandas as pd
    #  debug: /data/video_pack/debug/data/InternVid-10M-FLT-INFO-top10.jsonl
    #  real: /data/streamlit_source/raw_json/InternVid-10M-FLT-INFO.jsonl
    meta_data = pd.read_json('/data/streamlit_source/raw_json/InternVid-10M-FLT-INFO.jsonl', lines=True) 
    print("Loaded internvid json")
    data = []
    
    for idx in tqdm(range(len(meta_data)), total=len(meta_data), desc='Converting internvid format to required format...'):
        file_name = os.path.join("_".join([meta_data['YoutubeID'][idx],meta_data['Start_timestamp'][idx],meta_data['End_timestamp'][idx]]) + ".mp4")
        caption = meta_data['Caption'][idx]
        
        data.append({
            'video_path': os.path.join(get_prefix('internvid'), file_name),
            'value': caption
        })
    
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

def load_llava(json_path="/mnt/shared-storage/tenant/hypertext/danielyu/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"):
    meta_data = json.load(open(json_path))
    data = []
    for i in tqdm(meta_data, total=len(meta_data), desc='Converting llava format to required format...'):
        data.append({
            'video_path': os.path.join(data_prefix['llava_pretrain'], i['image']),
            'value': i['conversations']
        })
    return data

def load_merlin(interleave_path):
    # This function is for merlin-s dataset
    if not os.path.exists(interleave_path):
        raise ValueError(f"interleave file {interleave_path} does not exist")
    data = []
    with open(interleave_path, 'r') as f:
        for record in tqdm(json.items(f, "item"), desc="processing the merlin-s dataset..."):
            data.append({
                'video_path': record['image_info'],
                'value': record['text_list']
            })
    return data

def find_files(root_dir, filename):
    matched_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filename in filenames:
            matched_files.append(os.path.join(dirpath, filename))
    return matched_files

def load_videochat2(json_file = "/mnt/shared-storage/tenant/hypertext/danielyu/data/VideoChat2/sample_100k_multi/videochat2_full_667k.json"):
    if not os.path.exists(json_file):
        raise ValueError(f"interleave file {json_file} does not exist")
    result = []
    datas = json.load(open(json_file))
    for data in tqdm(datas, total=len(datas)):
        result.append({
            'images': data['images'],
            'conversations': data['conversations']
        })
    return result