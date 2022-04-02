import json
import transmission
from frame_utils import Frame
import frame_utils


METADATA_MESSAGE = 0
METADATA_ACK_MESSAGE = 1
IMAGE_ACK_MESSAGE = 2


class FrameTransmissionReceiver(transmission.TransmissionReceiver):
    def handle_data(self, sock, data):
        txt = data.decode()
        message = json.loads(txt)
        if 'type' in message:
            if message['type'] == METADATA_MESSAGE:
                image_size = message['size']
                bytes_size = message['bytes_size']
                frame_seq = message['seq']
                size_ack_message = {'type': METADATA_ACK_MESSAGE}
                sock.sendall(json.dumps(size_ack_message).encode())

                # read the image
                image_bytes = frame_utils.socket_recv_all(sock, bytes_size)
                if not image_bytes:
                    raise Exception('No image received.')
                image_ack_message = {'type': IMAGE_ACK_MESSAGE}
                sock.sendall(json.dumps(image_ack_message).encode())
                image = frame_utils.bytes_to_image(image_bytes)
                frame = Frame(image, image_size, frame_seq)
                self.queue.put(frame)
            else:
                raise Exception(f'Not expecting message of {message["type"]}.')
        else:
            raise Exception('Message has no type.')


class FrameTransmissionSender(transmission.TransmissionSender):
    def send(self, frame):
        print(f'Sending frame {frame.frame_seq}...')
        frame_bytes = frame_utils.image_to_bytes(frame.image)
        metadata_message = {
            'type': METADATA_MESSAGE,
            'size': frame.size,
            'bytes_size': len(frame_bytes),
            'seq': frame.frame_seq
        }
        self.sender_socket.sendall(json.dumps(metadata_message).encode())
        print('Sending frame metadata...')
        metadata_ack = json.loads(self.sender_socket.recv(transmission.BUFFER_SIZE))
        if metadata_ack is None or \
                'type' not in metadata_ack or \
                metadata_ack['type'] != METADATA_ACK_MESSAGE:
            raise Exception(f'Invalid message {metadata_ack}.')
        print('Metadata ack received.')
        print('Sending the image...')
        self.sender_socket.sendall(frame_bytes)
        image_ack = json.loads(self.sender_socket.recv(transmission.BUFFER_SIZE))
        if image_ack is None or \
                'type' not in image_ack or \
                image_ack['type'] != IMAGE_ACK_MESSAGE:
            raise Exception(f'Invalid message {image_ack}.')
        print('Image ack received.')
