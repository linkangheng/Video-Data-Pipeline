该仓库构建一个完整的视频数据处理pipeline，涉及视频抽帧、视频打包、数据tokenized、数据检查等部分。
### TIP:
1. Parallel 库不支持 ffmpeg, 换成了 multiprocessing 库