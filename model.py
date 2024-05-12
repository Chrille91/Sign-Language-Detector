import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

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

