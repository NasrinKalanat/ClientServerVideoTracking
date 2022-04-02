import argparse
from frame_transmission import FrameTransmissionReceiver
from object_transmission import ObjectTransmissionSender, DetectedObjects, DetectedObject
from queue import Queue
import threading
import time
import torchvision
import torchvision.transforms as T
import frame_utils
from frame_utils import Frame


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class ObjectDetectionServer:
    def __init__(self, host, port, client_host, client_port):
        self.host = host
        self.port = port
        self.client_host = client_host
        self.client_port = client_port
        self.frame_queue = Queue()
        self.frame_receiver = FrameTransmissionReceiver(self.host, self.port, self.frame_queue)
        self.object_sender = ObjectTransmissionSender(self.client_host, self.client_port)
        self.model = None
        self.connection_thread = None
        self.server_thread = None

    def start(self):
        self.frame_receiver.start()
        print('Waiting for client...')
        while True:
            if self.object_sender.connect():
                print('Connected to client.')
                break
            time.sleep(2)
        self.model = self.load_model()
        self.server_thread = threading.Thread(target=self.server_loop)
        self.server_thread.start()
        self.server_thread.join()

    def load_model(self):
        # load model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # set to evaluation mode
        model.eval()
        return model

    def detect_object(self, frame: Frame):
        transform = T.Compose([T.ToTensor()])
        img = frame_utils.cv2_to_pil(frame.image)
        img = transform(img)
        preds = self.model([img])
        objects = []
        for label, bbox, score in list(zip(
                preds[0]['labels'].numpy(),
                preds[0]['boxes'].detach().numpy(),
                preds[0]['scores'].detach().numpy())):
            bbox = [float(item) for item in bbox]
            detected_object = DetectedObject(int(label), bbox, float(score))
            objects.append(detected_object)
        return DetectedObjects(frame.frame_seq, objects)

    def server_loop(self):
        print('Server loop started')
        while True:
            try:
                max_seq = -1
                max_frame = None
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    if frame.frame_seq > max_seq:
                        max_frame = frame
                        max_seq = frame.frame_seq
                frame = max_frame
                if frame is None:
                    time.sleep(0.1)
                    continue
                print(f'Frame #{frame.frame_seq} received.')
                objects = self.detect_object(frame)
                print(f'{len(objects.objects)} objects detected.')
                self.object_sender.send(objects)
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as ex:
                print(ex)


# server program arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Server process.')
    parser.add_argument('-host', type=str, required=True, help='Server host')
    parser.add_argument('-port', type=int, required=True, help='Server port')
    parser.add_argument('-client_host', type=str, required=True, help='Client host')
    parser.add_argument('-client_port', type=int, required=True, help='Client port')
    return parser


if __name__ == '__main__':
    try:
        arg_parser = build_arg_parser()
        args = arg_parser.parse_args()  # parse arguments
        server = ObjectDetectionServer(args.host, args.port, args.client_host, args.client_port)
        server.start()
    except Exception as e:
        print(e)
