import numpy as np
import os

# actions = np.array(['Hi', 'ILoveYou', '_'])
# original value ['Hi', 'Yes', 'No', 'ThankYou', 'ILoveYou', 'background', 'NoHands']
# - not changed yet, just writing it down here since I think we need to have a more flexible solution for changing the list of names... (?)

actions = np.array(['Hi', 'Yes', 'No', 'ThankYou', 'ILoveYou', 'background', 'NoHands'])

facemesh_included = False

is_martin = False

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

# Model Parameters
neural_factor = 1 # used for naming of weights file, so only update number 
model_type = "LSTM"
activation_function = "tanh"
activation = "softmax"
neural_factor = neural_factor
metrics = ['accuracy', 'categorical_accuracy', 'Precision', 'Recall']