import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

from config import facemesh_included, number_of_classes
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# activation_function = "relu"
# activation_function = "tanh"


# original function before implementing the build_which_model function - leaving untouched so as Christian can usi it if he wants
def create_model(model_typ = "LSTM", act_funct = "tanh", activation = "softmax", neural_factor = 1):
    if model_typ == "LSTM":
        if facemesh_included == True:
            number_of_keypoints = 1662 # 258 if no facemesh, 1662 if facemesh is included
            coefficient = 1 # if coefficient is 1, the model structure is the same as it is in the original model by Nick
        elif facemesh_included == False:
            number_of_keypoints = 258 # 258 if no facemesh, 1662 if facemesh is included
            coefficient = 0.5 # coefficient 0.5 means that the number of neurons in 2nd, the 3rd and the 4th layer will be half - so as to account for the smaller input shape WHEN FACEMESH is REMOVED

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation=act_funct, input_shape=(30, number_of_keypoints)))
        model.add(LSTM(int(128*coefficient*neural_factor), return_sequences=True, activation=act_funct))
        model.add(LSTM(int(64*coefficient*neural_factor), return_sequences=False, activation=act_funct))
        model.add(Dense(int(64*coefficient*neural_factor), activation=act_funct))
        model.add(Dense(32*neural_factor, activation=act_funct))
        model.add(Dense(number_of_classes, activation=activation)) 

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model
    
    # adapted from Nick Renotte: https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb
    elif model_typ == "Conv2D":
        model.add(Conv2D(16, (3,3), 1, activation=act_funct, input_shape=(256,256,3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(32*coefficient*neural_factor, (3,3), 1, activation=act_funct))
        model.add(MaxPooling2D())
        model.add(Conv2D(16*coefficient*neural_factor, (3,3), 1, activation=act_funct))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256*neural_factor, activation=act_funct))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model



############################################################################################################

def build_which_model(model_name):
    # Access the function from the global scope
    model_function = globals().get(model_name)
    # Check if the function exists
    if model_function:
        return model_function()
    else:
        raise ValueError(f"No such model function: {model_name}")

# Christians default model
def model_1(model_typ = "LSTM", act_funct = "tanh", activation = "softmax", neural_factor = 1):
    if model_typ == "LSTM":
        if facemesh_included == True:
            number_of_keypoints = 1662 # 258 if no facemesh, 1662 if facemesh is included
            coefficient = 1 # if coefficient is 1, the model structure is the same as it is in the original model by Nick
        elif facemesh_included == False:
            number_of_keypoints = 258 # 258 if no facemesh, 1662 if facemesh is included
            coefficient = 0.5 # coefficient 0.5 means that the number of neurons in 2nd, the 3rd and the 4th layer will be half - so as to account for the smaller input shape WHEN FACEMESH is REMOVED

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation=act_funct, input_shape=(30, number_of_keypoints)))
        model.add(LSTM(int(128*coefficient*neural_factor), return_sequences=True, activation=act_funct))
        model.add(LSTM(int(64*coefficient*neural_factor), return_sequences=False, activation=act_funct))
        model.add(Dense(int(64*coefficient*neural_factor), activation=act_funct))
        model.add(Dense(32*neural_factor, activation=act_funct))
        model.add(Dense(number_of_classes, activation='softmax')) 

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model
    
    # adapted from Nick Renotte: https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb
    elif model_typ == "Conv2D":
        model.add(Conv2D(16, (3,3), 1, activation=act_funct, input_shape=(256,256,3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(32*coefficient*neural_factor, (3,3), 1, activation=act_funct))
        model.add(MaxPooling2D())
        model.add(Conv2D(16*coefficient*neural_factor, (3,3), 1, activation=act_funct))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256*neural_factor, activation=act_funct))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model

# first M's model taken from his last notebook - adjusted for the number of classes in the config.py
# ['Hi', 'Yes', 'No', 'ThankYou', 'ILoveYou', 'background', 'NoHands']
def model_19():
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.regularizers import l2

    # activation_function = "relu"
    activation_function = "tanh"

    actions = np.array(['Hi', 
                        'Yes', 
                        'No', 
                        'ThankYou', 
                        'ILoveYou', 
                        'background', 
                        'NoHands'
                        ])

    class_weights = {
        0: 1.0, 
        1: 1.0, 
        2: 1.0,  
        3: 1.0,
        4: 1.50, # I Love You
        5: 2.0, 
        6: 2.0,# _
    }

    if facemesh_included:
            coeficient = 1
            number_of_keypoints = 1662
    else:
            number_of_keypoints = 258
            coeficient = 1

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 1662), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(number_of_classes, activation='softmax'))


    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model