from abc import ABC, abstractmethod
import select
import socket
from queue import Queue
import threading


BUFFER_SIZE = 4096


class TransmissionReceiver(ABC):
    def __init__(self, host, port, queue: Queue):
        self.host = host
        self.port = port
        self.queue = queue
        self.connected_clients_sockets = []
        self.server_socket = None
        self.receiver_thread = None
        self.stop = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        self.connected_clients_sockets.append(self.server_socket)
        self.receiver_thread = threading.Thread(target=self.receiving_loop)
        self.receiver_thread.start()

    def receiving_loop(self):
        while not self.stop:
            try:
                read_sockets, write_sockets, error_sockets = \
                    select.select(self.connected_clients_sockets, [], [])
                for sock in read_sockets:
                    if sock == self.server_socket:
                        client_socket, client_address = self.server_socket.accept()
                        self.connected_clients_sockets.append(client_socket)
                    else:
                        try:
                            data = sock.recv(BUFFER_SIZE)
                            self.handle_data(sock, data)
                        except (KeyboardInterrupt, SystemExit) as ex:
                            raise ex
                        except Exception as ex:
                            print(f'Error: {ex}')
                            continue
            except (KeyboardInterrupt, SystemExit):
                break

    @abstractmethod
    def handle_data(self, sock, data):
        raise NotImplementedError()

    def close(self):
        self.stop = True
        for sock in self.connected_clients_sockets:
            if sock == self.server_socket:
                sock.close()
            else:
                sock.shutdown()


class TransmissionSender(ABC):
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.sender_socket = None

    def connect(self):
        try:
            self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sender_socket.connect((self.server_host, self.server_port))
            return True
        except Exception as ex:
            print(ex)
            return False

    @abstractmethod
    def send(self, data):
        raise NotImplementedError()

    def close(self):
        self.sender_socket.close()
