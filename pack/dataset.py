from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import os
import re
from megfile import smart_open, smart_exists
import tempfile
import shutil

class videoItem():
    platform_endpoint = {
        "oss": "http://oss.i.basemind.com",
        "sos": "http://oss.i.shaipower.com",
        "tos-huadong-2": "http://tos-s3-cn-shanghai.ivolces.com",
        "tos-huabei-3": "http://tos-s3-cn-beijing2.ivolces.com"
    }

    def __init__(self, video_path, platform, num_segments=-1):
        self.platform = platform
        self.video_path = video_path
        self.video_online = True if "s3" in self.video_path else False
        self.num_segments = num_segments
        self.vr = self.get_vr()
        self.max_frame = len(self.vr) - 1
        self.duration = self.get_duration()
        self.fps = self.vr.get_avg_fps()

    def get_vr(self):
        if self.video_online:
            video_path = self.get_cache_video(self.video_path)
        else:
            video_path = self.video_path
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        if self.video_online:
            os.remove(video_path)
        return vr

    def get_duration(self):
        duration = len(self.vr) / self.vr.get_avg_fps()
        return duration

    def get_platform(self):
        try:
            with open("/kubebrain/authorized_keys.sh", "r") as f:
                content = f.read()
            if "basemind" in content:
                return "oss"
            elif "shaipower" in content:
                return "sos"
        except:
            pass

        ssh_env_file = os.path.join(os.environ.get('HOME'), ".ssh/environment")
        if os.path.exists(ssh_env_file):
            with open(ssh_env_file, "r") as f:
                content = f.read()
            match = re.search(r"MLP_REGION=([^\n]+)", content)
            if match:
                if "shanghai" in match.group(1):
                    return "tos-huadong-2"
                elif "beijing" in match.group(1):
                    return "tos-huabei-3"
        return "unknown-platform"

    def get_cache_video(self, video_path):
        os.environ['OSS_ENDPOINT'] = self.platform_endpoint[self.platform]

        if not smart_exists(video_path):
            error_message = f"Video file not found: {video_path}"
            if smart_exists(video_path.replace("process_videos", "videos")):
                video_path = video_path.replace("process_videos", "videos")
            else:
                raise FileNotFoundError(error_message)

        with smart_open(video_path, 'rb') as file_obj:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                shutil.copyfileobj(file_obj, temp_file)
                cache_video_path = temp_file.name

        return cache_video_path

    def get_index(self, bound, first_idx=0, num_segments=-1):
        # use for sampling
        if not bound and num_segments < 0:
            num_segments = self.num_segments
        else:
            AssertionError("num_segments must be set when bound is set!")

        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * self.fps))
        end_idx = min(round(end * self.fps), self.max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def read_video(self, bound=None, num_segments = -1):
        images_group = list()
        frame_indices = self.get_index(bound, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(self.vr[frame_index].asnumpy())
            images_group.append(img)
        return images_group

    def get_video_clip(self, start, end, num_segments):
        assert num_segments > 0, "num_segments must be greater than 0"
        return self.read_video([start, end], num_segments)

    def get_relative_timestamp(self, abs_timestamp, time_scales = 1000):
        relative_timestamp = round((abs_timestamp / self.duration) * time_scales, 1)
        return relative_timestamp

    def __str__(self):
        return f"video_path: {self.video_path}, platform: {self.platform}"

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    video_path = "s3://kanelin/video_data/Momentor/video/3HRE5JLV9_Y.mp4"
    platform = "oss"
    video = videoItem(video_path, platform, num_segments=16)
