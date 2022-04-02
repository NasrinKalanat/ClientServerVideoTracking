import numpy as np
import torch

from tracker.network import Re3Net
import utils.bb_util as bb_util
import utils.im_util as im_util

CROP_SIZE = 227
CROP_PAD = 2

MAX_TRACK_LENGTH = 4


class Re3Tracker(object):
    def __init__(self, model_path='checkpoint.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Re3Net().to(self.device)
        if model_path is not None:
            if self.device.type == "cpu":
                self.net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            else:
                self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.tracked_data = {}

    def track(self, id, image, bbox=None):
        image = image.copy()

        if bbox is not None:
            lstm_state = None
            past_bbox = bbox
            prev_image = image
            forward_count = 0
        elif id in self.tracked_data:
            lstm_state, initial_state, past_bbox, prev_image, forward_count = self.tracked_data[id]
        else:
            raise Exception('Id {0} without initial bounding box'.format(id))

        cropped_input0, past_bbox_padded = im_util.get_cropped_input(prev_image, past_bbox, CROP_PAD, CROP_SIZE)
        cropped_input1, _ = im_util.get_cropped_input(image, past_bbox, CROP_PAD, CROP_SIZE)

        network_input = np.stack((cropped_input0.transpose(2, 0, 1), cropped_input1.transpose(2, 0, 1)))
        network_input = torch.tensor(network_input, dtype=torch.float)

        with torch.no_grad():
            network_input = network_input.to(self.device)
            network_predicted_bbox, lstm_state = self.net(network_input, prevLstmState=lstm_state)

        if forward_count == 0:
            initial_state = lstm_state
            # initial_state = None

        prev_image = image

        predicted_bbox = bb_util.from_crop_coordinate_system(network_predicted_bbox.cpu().data.numpy() / 10,
                                                             past_bbox_padded, 1, 1)

        # Reset state
        if forward_count > 0 and forward_count % MAX_TRACK_LENGTH == 0:
            lstm_state = initial_state

        forward_count += 1

        if bbox is not None:
            predicted_bbox = bbox

        predicted_bbox = predicted_bbox.reshape(4)

        self.tracked_data[id] = (lstm_state, initial_state, predicted_bbox, prev_image, forward_count)

        return predicted_bbox

    def reset(self):
        self.tracked_data = {}
