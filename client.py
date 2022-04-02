import argparse
from queue import Queue
from frame_transmission import FrameTransmissionSender
from object_transmission import ObjectTransmissionReceiver, DetectedObjects, DetectedObject
import threading
import time
import frame_utils
from frame_utils import Frame
from tracker.re3_tracker import Re3Tracker
import cv2
import os


MAX_OBJ_TRACK_NUM = 1
FRAME_DIFF_THRESHOLD = 4500000
MIN_KEY_FRAME_DISTANCE = 0.1    # seconds


class ObjectTrackerClient:
    def __init__(self, host, port, server_host, server_port, video_path, frame_rate, output):
        self.host = host
        self.port = port
        self.server_host = server_host
        self.server_port = server_port
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.output_path = output
        self.video_buffer = Queue()
        self.object_queue = Queue()
        self.frame_sender = FrameTransmissionSender(self.server_host, self.server_port)
        self.object_receiver = ObjectTransmissionReceiver(self.host, self.port, self.object_queue)
        self.client_thread = None
        self.video_thread = None
        self.trackers = None

    def start(self):
        self.object_receiver.start()
        print('Waiting for server...')
        while True:
            if self.frame_sender.connect():
                print('Connected to server.')
                break
            time.sleep(2)
        self.client_thread = threading.Thread(target=self.client_loop)
        self.client_thread.start()
        self.video_thread = threading.Thread(target=self.video_sim_loop)
        self.video_thread.start()
        self.client_thread.join()

    def client_loop(self):
        first_frame: Frame = self.video_buffer.get()
        self.frame_sender.send(first_frame)
        first_objects: DetectedObjects = self.object_queue.get()
        self.trackers = [Re3Tracker() for _ in range(MAX_OBJ_TRACK_NUM)]
        for i, tracker in enumerate(self.trackers):
            tracker.track(1, first_frame.image, bbox=first_objects.objects[i].bbox)
        last_frame_sent = first_frame
        frame_cnt = 0
        frame_cache = []
        sent_frames = {}
        while True:
            try:
                frame: Frame = self.video_buffer.get()
                frame_cnt += 1
                diff = frame_utils.diff_img(last_frame_sent.image, frame.image)
                print(f'diff between #{last_frame_sent.frame_seq} and #{frame.frame_seq} is {diff}')
                if diff > FRAME_DIFF_THRESHOLD and frame_cnt * (1 / self.frame_rate) > MIN_KEY_FRAME_DISTANCE:
                    # send to server
                    frame_cnt = 0
                    self.frame_sender.send(frame)
                    sent_frames[frame.frame_seq] = frame
                    last_frame_sent = frame
                if not self.object_queue.empty():   # reset the trackers
                    max_seq = -1
                    objects = None
                    key_frame = None
                    while not self.object_queue.empty():
                        obj = self.object_queue.get()
                        if max_seq < obj.frame_seq:
                            max_seq = obj.frame_seq
                            objects = obj
                            key_frame = sent_frames[obj.frame_seq]
                        del sent_frames[obj.frame_seq]
                    for i, tracker in enumerate(self.trackers):
                        tracker.track(1, key_frame.image, bbox=objects.objects[i].bbox)
                    for cached_frame in frame_cache:
                        if cached_frame.frame_seq <= objects.frame_seq:
                            continue
                        for i, tracker in enumerate(self.trackers):
                            tracker.track(1, cached_frame.image)
                    frame_cache.clear()
                else:
                    frame_cache.append(frame)
                    frame_image = frame.image.copy()
                    for i, tracker in enumerate(self.trackers):
                        bbox = tracker.track(1, first_frame.image)
                        frame_image = frame_utils.draw_bbox(frame_image, bbox, i)
                    cv2.imwrite(os.path.join(self.output_path, f'{str(frame.frame_seq)}.JPEG'), frame_image)
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as ex:
                pass

    def video_sim_loop(self):
        video_reader = frame_utils.DirectoryVideoReader(self.video_path)
        for frame in frame_utils.video_stream(video_reader, self.frame_rate):
            self.video_buffer.put(frame)


# client program arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Server process.')
    parser.add_argument('-host', type=str, required=True, help='Client host')
    parser.add_argument('-port', type=int, required=True, help='Client port')
    parser.add_argument('-server_host', type=str, required=True, help='Server host')
    parser.add_argument('-server_port', type=int, required=True, help='Server port')
    parser.add_argument('-video_path', type=str, required=True, help='Video Path')
    parser.add_argument('-frame_rate', type=int, required=True, help='Video FPS')
    parser.add_argument('-out', type=str, required=True, help='Output Path')
    return parser


if __name__ == '__main__':
    try:
        arg_parser = build_arg_parser()
        args = arg_parser.parse_args()  # parse arguments
        object_tracker = ObjectTrackerClient(args.host, args.port, args.server_host, args.server_port, args.video_path,
                                             args.frame_rate, args.out)
        object_tracker.start()
    except Exception as e:
        print(e)
