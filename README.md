本仓库旨在构建一个完整的视频数据处理pipeline，主要用于预处理视频训练数据，涉及视频抽帧、视频数据打包、数据tokenized、数据检查等部分。

video data pipeline 一般面向两个需求，一是支持megatron的训练，二是支持使用webdataset进行data loading 的各类模型训练。

# Pipeline:

## Megatron

Megatron 的视频数据预处理目前的总体思想是：

**将一切预处理（包括图像处理、数据拼接、以及数据 token 化）都离线完成**。

因此，当前数据处理 pipeline 分为以下三个阶段：

1. **原始数据（pair & interleave）打包**（从各类离散存储方式转化为 webdataset 可读的 tarfiles，图片 & 文本信息全部存储在 tarfile），对应 **`Video-Data-Pipeline/pack.py`**
2. **多个单点数据拼接到 8k 并完成 tokenization**（将单点数据转化为长度为 8k 的 webdataset 可读的 tarfiles，同时存储 input_ids & loss_mask）, 对应 **`Video-Data-Pipeline/tokenize/main.py`**
3. **数据统计检查**（将所有【tarfile url，tarfile 内数据量】存储在 pickle 中，并在统计过程中完成数据验证和检查）, 对应 **`Video-Data-Pipeline/index/main.py`**

## Webdataset

只需完成megatron pipeline 的第一步

# Data Format:

Video - Text Pair
数据格式：
- 500 samples / tar
- 多机多线程并行
- Tar包写入格式：
  - shard-machine_id-sample_start-sample_end-tar_id.tar
    - 000000001.json
    - 000000001.mp4
  - 参数说明：
    - machine_id：负责负责该tar包的机器id
    - sample_start-sample_end: 处理该tar包的线程所负责的sample范围
    - tar_id：该tar在其所负责的线程中的id
Examples:
- Tar 包结构
  ![image](https://github.com/linkangheng/Video-Data-Pipeline/assets/90882794/413d8dda-eb02-4952-b61e-107a9d0c8267)
  
- Json 内容：
{"caption": "a girl is talking to another girl sitting in a park", "video_id": "000000001"}
- Webdataset key:
  - mp4: 视频的二进制文件
  - json: 视频信息

# Usage:

## **Video Packing:**

1. code path: **`Video-Data-Pipeline/pack.py`**
  
2. 使用前需要先安装**`Video-Data-Pipeline/webdataset-private` ,** 这是组里修改过的webdataset，直接pip install webdataset 使用该仓库将会报错
  
3. 加载不同数据集时，约定将各类数据集统一成以下format：
  
  ```
  dataset = [
         {
             "video_path": video_path,
             "value": caption/other info
         },
         {
             ...
         },
         ...
     ]
  ```
  
  具体可以参考下面的函数进行修改，请保持字典的key不变，遇到 oss/tos 上的数据可以使用**`load_image`**函数进行在线读取：
  
  ```python
  def load_webvid():
     os.environ['OSS_ENDPOINT'] = '<http://oss.i.basemind.com>'
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
  ```
  
4. 代码目前支持**`how2link`**, **`webvid`**, **`hd-vila`**, **`ego4d`** 等视频数据集
  
5. 代码支持多种视频打包处理方案，包括完整视频打包(video_only), 均匀抽帧打包(un), 关键帧抽帧打包(kf)等, 方法写在samplers.py中，使用时请指定 --type 参数选择是否对视频进行抽帧以及抽帧的方案，若需要新的视频处理可以直接添加到samplers.py ，format可以参考下面的code
  
  ```python
  def video_processor(file_idx, images, args=None):
     image_name_list = []
     image_dict_list = []
  
     for index, image in enumerate(images):
         # 对图像或视频进行处理...
         processed_image = processor(image)
  
         # 最后需要把处理好的图像整理成两个list并返回
         image_name_list.append(f"{file_idx:09d}-{index}")
         image_dict_list.append(dict(
             __key__=f"{file_idx:09d}-{index}",
             jpg=img,
         ))
  
     return image_name_list, image_dict_list
  ```
  
6. examples scripts:
  
  python [pack.py](http://pack.py) --machine_id 0 --total_machine 8 --dataset internvid --workers 64 --type video-only --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/video_pack/video_only/internvid
  

## **Tokenization:**

1. code path: **`Video-Data-Pipeline/tokenize/main.py`**
  
2. 代码目前只支持了LLAMA2 作为tokenizer，有别的需求可以自行替换
  
3. examples scripts:
  
  ```python
  cd /data/video_pack/tokenize
  
  # start 和 end 需要根据第一步pack得到的tar包数以及并行的机器数来定
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 0 --end 1500 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 1500 --end 3000 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 3000 --end 4500 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 4500 --end 6000 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 6000 --end 7500 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 7500 --end 9000 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 9000 --end 10500 --sample_type kf
  python main.py --dataset_name howtointerlink-kf-v1 --workers 64 --shard_size 5 --start 10500 --end 12288 --sample_type kf
  ```
  

## Pair Check

```bash
conda activate webvid
cd /data/video_pack/index

python check_pair.py --dataset_name webvid-un-v1 --workers 64
python check_pair.py --dataset_name webvid-kf-v1 --workers 64
python check_pair.py --dataset_name hd3m-un-v1 --workers 64
python check_pair.py --dataset_name hd3m-kf-v1 --workers 64
python check_pair.py --dataset_name internvid-un-v1 --workers 64
python check_pair.py --dataset_name internvid-kf-v1 --workers 64
python check_pair.py --dataset_name howtointerlink-kf-v1 --workers 64
```
