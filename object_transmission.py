import json
import transmission
import numpy as np


class DetectedObject:
    def __init__(self, label, bbox, confidence):
        self.label = label
        self.bbox = bbox
        self.confidence = confidence

    def to_json(self):
        return json.dumps({
            'label': self.label,
            'bbox': self.bbox,
            'confidence': self.confidence
        })

    @staticmethod
    def from_dict(json_obj):
        object_dic = json.loads(json_obj)
        detected_object = DetectedObject(
            object_dic['label'],
            np.asarray(object_dic['bbox']),
            object_dic['confidence'])
        return detected_object


class DetectedObjects:
    def __init__(self, frame_seq, objects=None):
        self.frame_seq = frame_seq
        if objects is None:
            objects = []
        self.objects = objects

    def to_json(self):
        return json.dumps({
            'seq': self.frame_seq,
            'objects': [obj.to_json() for obj in self.objects],
        })

    @staticmethod
    def from_json(json_str):
        objects_dic = json.loads(json_str)
        objects = [DetectedObject.from_dict(obj) for obj in objects_dic['objects']]
        detected_objects = DetectedObjects(objects_dic['seq'], objects)
        return detected_objects


class ObjectTransmissionReceiver(transmission.TransmissionReceiver):
    def handle_data(self, sock, data):
        txt = data.decode()
        detected_objects = DetectedObjects.from_json(txt)
        self.queue.put(detected_objects)
        sock.sendall('ack'.encode())


class ObjectTransmissionSender(transmission.TransmissionSender):
    def send(self, detected_objects):
        detected_objects.objects.sort(key=lambda obj: obj.confidence, reverse=True)
        print(f'Sending object {detected_objects.frame_seq}...')
        object_message = detected_objects.to_json().encode()
        self.sender_socket.sendall(object_message)
        ack = self.sender_socket.recv(transmission.BUFFER_SIZE).decode()
        if ack and ack == 'ack':
            print('Object ack received.')
        else:
            raise Exception(f'Invalid message {ack}.')
