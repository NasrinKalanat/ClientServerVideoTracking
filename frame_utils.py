import io
import cv2
from PIL import Image
from abc import ABC, abstractmethod
import numpy as np
import os
import time
import utils.bb_util as bb_util
import random


COLOR_NUM = 100
RANDOM_COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(COLOR_NUM)]


def image_to_bytes(image):
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    if is_success:
        return im_buf_arr.tobytes()
    return None


def bytes_to_image(image_bytes):
    nparr = np.frombuffer(image_bytes, dtype="uint8")
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def pil_to_cv2(image):
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def cv2_to_pil(open_cv_image):
    return Image.fromarray(open_cv_image)


def socket_recv_all(sock, size):
    chunk_size = 2048
    data = b""
    while True:
        chunk = sock.recv(chunk_size)
        if not chunk:
            break
        data += chunk
        if len(data) == size:
            break
    return data


def diff_img(img1, img2):
    # Grey and resize
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1 = cv2.resize(img1, (320, 200), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (320, 200), interpolation=cv2.INTER_AREA)
    # Calculate
    return (abs(img2 - img1)).sum()


# BBoxes are [x1 y1 x2 y2]
def draw_bbox(image, bbox, color_id=0, padding=1):
    color = RANDOM_COLORS[color_id % COLOR_NUM]
    image_height = image.shape[0]
    image_width = image.shape[1]
    bbox = np.round(np.array(bbox))  # mostly just for copying
    bbox = bb_util.clip_bbox(bbox, padding, image_width - padding, image_height - padding).astype(int).squeeze()
    padding = int(padding)
    image[bbox[1] - padding:bbox[3] + padding + 1, bbox[0] - padding:bbox[0] + padding + 1] = color
    image[bbox[1] - padding:bbox[3] + padding + 1, bbox[2] - padding:bbox[2] + padding + 1] = color
    image[bbox[1] - padding:bbox[1] + padding + 1, bbox[0] - padding:bbox[2] + padding + 1] = color
    image[bbox[3] - padding:bbox[3] + padding + 1, bbox[0] - padding:bbox[2] + padding + 1] = color
    return image


class Frame:
    def __init__(self, image, size, frame_seq):
        self.image = image
        self.size = size
        self.frame_seq = frame_seq


class VideoReader(ABC):
    @abstractmethod
    def get_frame(self, frame_seq):
        raise NotImplementedError()

    @abstractmethod
    def next_frame(self):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self):
        raise NotImplementedError()


class DirectoryVideoReader(VideoReader):
    def __init__(self, directory_path, start_seq=0):
        self.start_seq = start_seq
        self.current_frame_seq = self.start_seq
        self.directory_path = directory_path
        self.frames_path = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path)]
        self.frames_path.sort()
        to_delete = []
        for path in self.frames_path:
            if path[0] == '.':
                to_delete.append(path)
                continue
            image = None
            try:
                image = cv2.imread(path)
            except:
                to_delete.append(path)
                continue
            finally:
                if image is not None:
                    del image
                else:
                    to_delete.append(path)
                    continue
        for path in to_delete:
            self.frames_path.remove(path)

    def get_frame(self, frame_seq):
        frame_path = self.frames_path[frame_seq]
        image = cv2.imread(frame_path)
        return Frame(image, list(image.shape), frame_seq)

    def next_frame(self):
        frame = self.get_frame(self.current_frame_seq)
        self.current_frame_seq += 1
        return frame

    def has_next(self):
        return self.current_frame_seq < len(self.frames_path)


def video_stream(video_reader: VideoReader, frame_rate=30):
    while video_reader.has_next():
        try:
            yield video_reader.next_frame()
            time.sleep(1 / frame_rate)
        except (KeyboardInterrupt, SystemExit):
            break
