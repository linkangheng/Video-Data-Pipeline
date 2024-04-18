import cv2
import os
import shutil
import re
from megfile import smart_open, smart_exists, smart_sync, smart_remove, smart_glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import threading
import math
from tqdm import tqdm 
import numpy as np
import argparse
from PIL import Image
import subprocess
import threading

lock = threading.Lock()

def get_cache_video(video_path):
    # Determine if the video exists
    if not smart_exists(video_path):
        error_message = f"Video file not found: {video_path}"
        if smart_exists(video_path.replace("process_videos", "videos")):
            video_path = video_path.replace("process_videos", "videos")
        else:
            raise FileNotFoundError(error_message)
    
    # Caching the video
    with smart_open(video_path, 'rb') as file_obj:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            shutil.copyfileobj(file_obj, temp_file)
            cache_video_path = temp_file.name
    return cache_video_path

def get_video_total_frames(video_path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1', video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    total_frames = int(result.stdout.strip())
    return total_frames
    
def extract_frames(video_path):
    interval = 13
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    images = []
    if total_frames < (interval-1) * fps:
        frame_step = fps
    else:
        frame_step = (total_frames - 1) // (interval -2)

    frame_count = 0
    sample_num = 0
    success = True

    while success and frame_count < total_frames -1 :
        try:
            success, image = cap.read()
        
        except cv2.error as e:
            success = False
            
        if not success:
            # Unable to read video normally,so use the last frame of the original video as sample
            if frame_count < total_frames -1:
                success = True
                frame_count += 1
                continue
            else:
                return images
            # raise Exception(f"Error: Unable to read video {video_path} normally")
        if (frame_count % frame_step == 0 and frame_count != total_frames) or frame_count == 0 or frame_count == total_frames - 1:
            try:
                if sample_num > interval:
                    assert False, "sample_num over interval!"
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(image))
                sample_num += 1
            except:
                import ipdb
                ipdb.set_trace()
        frame_count += 1

    cap.release()
    return images

def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 2)
    ret, frame = cap.read()
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frame

def keyframes_sampler(video_path, frame_type, max_samples, args=None):
    '''
    input: video_path,type = 'I' or 'P'
    output: images,indices,total_frames
    '''
    # Extract all keyframes(Iframes) from the video, Only Support Intern-Vid now
    images = []
    total_frames = get_video_total_frames(video_path)
    
    frame_info = subprocess.check_output(['ffprobe', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=pict_type', '-of', 'csv', video_path],stderr=subprocess.DEVNULL).decode().strip().split('\n')
    temp_dir = tempfile.TemporaryDirectory()
    output_pattern = os.path.join(temp_dir.name, 'output_%d.jpg')
    if frame_type == 'I':
        indices = [i for i, line in enumerate(frame_info) if line.strip().endswith("I")]
        command = ['ffmpeg', '-hide_banner', '-i', video_path, '-vf', "select='eq(pict_type\,I)'", '-vsync', 'vfr', '-f', 'image2', output_pattern]
    elif frame_type == 'P':
        indices = [i for i, line in enumerate(frame_info) if not line.strip().endswith("I")]
        command = ['ffmpeg', '-hide_banner', '-i', video_path, '-vf', "select='not(eq(pict_type\,I))'", '-vsync', 'vfr', '-f', 'image2', output_pattern]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    temp_files = sorted([os.path.join(temp_dir.name, f) for f in os.listdir(temp_dir.name) if f.startswith('output_')])

    images = [Image.open(temp_file) for temp_file in temp_files]
    
    if frame_type == 'P':
        # 因为I帧会自动添加最后一帧，所以这里需要去掉P帧的最后一帧
        images = images[:-1]
        indices = indices[:-1]
    
    elif frame_type == 'I':
        last_frame = os.path.join(temp_dir.name,"last_frame.jpg")
        images.append(get_last_frame(video_path))
        indices.append(total_frames-1)

    images = uniform_sample(images, max_samples)
    indices = uniform_sample(indices, max_samples)
    
    temp_dir.cleanup()    

    return images,indices,total_frames

def uniform_sample(lst, num_samples):
    length = len(lst)
    if length <= num_samples:
        return lst
    
    interval = (length - 1) / (num_samples - 1)
    result = []
    for i in range(0, length, int(interval)):
        result.append(lst[i])
        if len(result) == num_samples:
            break
    return result

def combineKeyFrames(I_images, I_indices, P_images, P_indices):
    """
    按照时序合并 I 帧和 P 帧, 使得合并后的帧按照时序排列
    
    参数:
    I_images (list): I 帧图像列表
    I_indices (list): I 帧图像在视频中的索引列表
    P_images (list): P 帧图像列表
    P_indices (list): P 帧图像在视频中的索引列表
    
    返回:
    new_images (list): 新的图像列表
    new_indices (list): 新的索引列表
    frame_types (list): 记录每个位置的帧类型, 0 表示 I 帧, 1 表示 P 帧
    """
    new_images = []
    new_indices = []
    frame_types = []

    i, p = 0, 0
    while i < len(I_indices) and p < len(P_indices):
        if I_indices[i] < P_indices[p]:
            new_images.append(I_images[i])
            new_indices.append(I_indices[i])
            frame_types.append("I")  # 0 表示 I 帧
            i += 1
        else:
            new_images.append(P_images[p])
            new_indices.append(P_indices[p])
            frame_types.append("P")  # 1 表示 P 帧
            p += 1

    # 如果还有剩余的I帧或P帧,则添加到新的list中
    if i < len(I_indices):
        new_images.extend(I_images[i:])
        new_indices.extend(I_indices[i:])
        frame_types.extend(["I"] * (len(I_indices) - i))
    elif p < len(P_indices):
        new_images.extend(P_images[p:])
        new_indices.extend(P_indices[p:])
        frame_types.extend(["P"] * (len(P_indices) - p))

    return new_images, new_indices, frame_types


def get_processed(log_file):
    try:
        with open(log_file,"r") as log:
            return len(log.readlines())
    except FileNotFoundError:
        return 0

def load_lines(lines_txt,prefix=""):
    if len(prefix)!=0:
        lines_txt = prefix + lines_txt
    lines = []
    with open(lines_txt,'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines 

def sample_process(local_video_path,s3_video_path):
    output_folder = s3_video_path.replace("s3://kanelin/interlink7m", "/data/howtolink/samples")[:-4] # for tos, output_folder = s3_video_path.replace("s3://kanelin/interlink7m", "/data/hypertext/kangheng/howto100m/samples")[:-4] 
    extract_frames(local_video_path,output_folder=output_folder)

def tracking_process(local_video_path):
    # TODO track all the videos 
    pass



def process_video(video_path):
    global delete_counter
    global cached_files
    global progress_count
    
    try:
        cache_video_path = get_cache_video(video_path)
    except Exception as e:
        error_message = f"Error caching file {video_path}: {e}"
        print(error_message)
        return
    
    # Process the video
    try:
        sample_process(cache_video_path,video_path)
        with threading.Lock():
            progress_count += 1
            video = "/".join(video_path.split("/")[-2:])
            progress_message = f"Processed video {progress_count}/{total_videos_count} ({video})"
            log(progress_message,progress_log_path)
            
    except Exception as e:
        error_message = f"Error processing file {video_path}: {e}"
        print(error_message)

    # Delete the cache
    cached_files.append(cache_video_path)
    delete_counter += 1
    
    if delete_counter >= DELETE_THRESHOLD:
        for cached_file in cached_files:
            os.remove(cached_file)
        delete_counter = 0
        cached_files = []
    
    
def main():
    global progress_count
    # Get the number of processed video
    progress_count = get_processed(progress_log_path)
    
    lines_file = "/data/hypertext/kangheng/project/merlin_track/videos/txt/" + TYPE + ".txt"
    lines = load_lines(lines_file) 
    lines = resume_lines(lines)

    for line in tqdm(lines,total=len(lines)):
        process_video(line)
        break
    # with ThreadPoolExecutor(max_workers) as executor:
    #     futures = [executor.submit(process_video, line) for line in lines]
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
    #         future.result()



def ditribute_main(machine_id):
    global progress_count
    global error_log_path
    global error_list_path
    global progress_log_path
    
    # Maintain logs based on machine ID for TOS
    log_prefix=os.path.join("/data/howtolink/result",TYPE) 
    error_log_path = os.path.join(log_prefix,str(machine_id)+"_error_details.txt")
    error_list_path = os.path.join(log_prefix,str(machine_id)+"_error_list.txt")
    progress_log_path = os.path.join(log_prefix,str(machine_id)+"_progress.txt")
    
    # Get the number of processed video
    progress_count = get_processed(progress_log_path)
    lines_file = "/data/howtolink/videos/txt/" + TYPE + ".txt" # for tos, lines_file = "/data/hypertext/kangheng/project/merlin_track/videos/txt/" + TYPE + ".txt" 
    lines = load_lines(lines_file)
    lines = get_devided_lines(lines,machine_id,interval=15)
    
    # for line in tqdm(lines,total=len(lines)):
    #     import time
    #     start = time.time()
    #     process_video(line)
    #     end = time.time()
    #     print("Process time: ",end-start)
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(process_video, line) for line in lines]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            future.result()

def debug():
    line = "s3://kanelin/interlink7m/Howto-Interlink7M_subset_w_all_clips_train/26n5ePOXc5I/clip_3.mp4"
    process_video(line)

if __name__ == "__main__":
    # global TYPE
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_id", type=int, default=0,help="Machine ID")
    # parser.add_argument("--type", type=str, help="Machine ID")
    args = parser.parse_args()
    machine_id = args.machine_id
    # TYPE = args.type
    ditribute_main(machine_id)
    # rclone ls oss:a-share/vd-foundation___InternVid-10M-FLT/raw/InternVId-FLT_11.zip