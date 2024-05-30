该仓库构建一个完整的视频数据处理pipeline，涉及视频抽帧、视频打包、数据tokenized、数据检查等部分。
## 使用说明：
### Video Packing:
1. 支持多种视频打包处理方案，包括直接视频打包(video_only mode), 均匀抽帧打包(un), 关键帧抽帧打包(kf)等, 方法写在samplers.py中
2. 针对不同数据集需要写一个加载数据集的函数load_dataset(), 将数据输入统一, 目前支持how2link, webvid, hd-vila, ego4d 等视频数据集
3. examples scripts:
   python pack.py --machine_id 0 --total_machine 8 --dataset internvid --workers 64 --type video-only --save_path /mnt/shared-storage/tenant/hypertext/kanelin/data/video_pack/video_only/internvid
