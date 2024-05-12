import numpy as np
import os

# actions = np.array(['Hi', 'ILoveYou', '_'])
actions = np.array(['Hi', 'Yes', 'No', 'ThankYou', 'ILoveYou', 'background', 'NoHands'])

facemesh_included = False

is_martin = True

# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('mediapipe_keypoints')
DATA_PATH = os.path.join('MP_Data')
# model weights folder; default: subfolder "weights"
WEIGHTS_PATH = os.path.join("weights")

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

number_of_classes = len(actions)