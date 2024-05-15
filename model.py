import os
import numpy as np
import config
import importlib
importlib.reload(config)
from config import actions, facemesh_included, number_of_classes

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Attention  # type: ignore
from tensorflow.keras.callbacks import TensorBoard  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout # type: ignore




# categorical_accuracy: This is a metric function that calculates the mean accuracy rate across all predictions for multi-class classification problems. It's used when your targets are one-hot encoded.
# accuracy: This calculates the mean accuracy rate across all predictions. It's used when your targets are binary or multilabel.
# Precision: This is the ratio of correctly predicted positive observations to the total predicted positives. High precision relates to the low false positive rate. It's used when the cost of False Positive is high.
# Recall (Sensitivity): This is the ratio of correctly predicted positive observations to the all observations in actual class. It's used when the cost of False Negative is high.
# ==> In our case, we want to make sure each sign is correctly recognized (high precision), and that we recognize as many instances of each sign as possible (high recall).


# original function before implementing the build_which_model function - leaving untouched so as Christian can usi it if he wants
def create_model(model_type = "LSTM", activation_function = "tanh", activation = "softmax", neural_factor = 1, metrics = ['categorical_accuracy', 'accuracy', 'Precision', 'Recall']):
    if model_type == "LSTM":
        if facemesh_included == True:
            number_of_keypoints = 1662 # 258 if no facemesh, 1662 if facemesh is included
            coefficient = 1 # if coefficient is 1, the model structure is the same as it is in the original model by Nick
        elif facemesh_included == False:
            number_of_keypoints = 258 # 258 if no facemesh, 1662 if facemesh is included
            coefficient = 0.5 # coefficient 0.5 means that the number of neurons in 2nd, the 3rd and the 4th layer will be half - so as to account for the smaller input shape WHEN FACEMESH is REMOVED

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation=activation_function, input_shape=(30, number_of_keypoints)))
        model.add(LSTM(int(128*coefficient*neural_factor), return_sequences=True, activation=activation_function))
        model.add(LSTM(int(64*coefficient*neural_factor), return_sequences=False, activation=activation_function))
        model.add(Dense(int(64*coefficient*neural_factor), activation=activation_function))
        model.add(Dense(32*neural_factor, activation=activation_function))
        model.add(Dense(number_of_classes, activation=activation)) 

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=metrics)
        print(metrics)

        return model
    
    # adapted from Nick Renotte: https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb
    # yet untested
    elif model_type == "Conv2D":
        model.add(Conv2D(16, (3,3), 1, activation=activation_function, input_shape=(256,256,3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(32*coefficient*neural_factor, (3,3), 1, activation=activation_function))
        model.add(MaxPooling2D())
        model.add(Conv2D(16*coefficient*neural_factor, (3,3), 1, activation=activation_function))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256*neural_factor, activation=activation_function))
        model.add(Dense(number_of_classes, activation='sigmoid'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

        return model


def create_improved_model(model_type="LSTM", activation_function="tanh", activation="softmax", neural_factor=1, metrics=['categorical_accuracy', 'accuracy', 'Precision', 'Recall']):
    # Determine the number of keypoints based on whether facemesh is included
    number_of_keypoints = 1662 if facemesh_included else 258
    
    # Coefficient to scale the number of neurons if facemesh is not included
    coefficient = 1 if facemesh_included else 0.5

    model = Sequential()
    
    # Adding a TimeDistributed Conv1D layer to extract spatial features from each frame
    # TimeDistributed allows applying the same Conv1D layer to each frame individually
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation=activation_function), input_shape=(30, number_of_keypoints, 1)))
    
    # Adding a TimeDistributed MaxPooling1D layer to reduce the spatial dimensions of each frame
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    
    # Adding a TimeDistributed Flatten layer to convert 2D features into 1D for each frame
    model.add(TimeDistributed(Flatten()))
    
    # Adding a Bidirectional LSTM layer to capture dependencies in both forward and backward directions
    # LSTM layer with 64 units, return_sequences=True to output a sequence of the same length
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation=activation_function)))
    
    # Adding another Bidirectional LSTM layer with more units for capturing more complex patterns
    # Number of units is scaled by coefficient to adjust for facemesh inclusion/exclusion
    model.add(Bidirectional(LSTM(int(128 * coefficient * neural_factor), return_sequences=True, activation=activation_function)))
    
    # Adding an Attention layer to focus on important frames or parts of the frames
    model.add(Attention())
    
    # Adding a Dense layer with activation function to introduce non-linearity
    # Number of units is scaled by coefficient to adjust for facemesh inclusion/exclusion
    model.add(Dense(int(64 * coefficient * neural_factor), activation=activation_function))
    
    # Adding another Dense layer with fewer units
    model.add(Dense(32 * neural_factor, activation=activation_function))
    
    # Output layer with softmax activation for multi-class classification
    model.add(Dense(number_of_classes, activation=activation)) 

    # Compiling the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=metrics)
    print(metrics)

    return model



############################################################################################################
# Martin's model

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
    from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
    from tensorflow.keras.regularizers import l2 # type: ignore

    # activation_function = "relu"
    activation_function = "tanh"

    # actions = np.array(['Hi', 
    #                     'Yes', 
    #                     'No', 
    #                     'ThankYou', 
    #                     'ILoveYou', 
    #                     'background', 
    #                     'NoHands'
    #                     ])

    empty_signs = ["background", "NoHands"]
    actions = [
        "ILoveYou",
        "Yes",
        "No",
        "Hi",
        "ThankYou",
        "Me",
        "You",
        "It",
        "Feel",
        "Happy",
        "Hungry",
        "Eat",
        "Bread",
        "Chocolate",
        "Tired",
        ]
    actions = empty_signs + actions

    # class_weights = {i: 1.0 for i in range(number_of_classes)}
    # class_weights = {
    #     0: 1.0, 
    #     1: 1.0, 
    #     2: 1.0,  
    #     3: 1.0,
    #     4: 1.50, # I Love You
    #     5: 2.0, 
    #     6: 2.0,# _
    # }
    class_weights = {
        0: 2.0,  # background
        1: 2.0,  # NoHands
        2: 1.0,  # ILoveYou
        3: 1.0,  # Yes
        4: 1.0,  # No
        5: 1.0,  # Hi
        6: 1.50, # ThankYou (adjusted weight)
        7: 1.0,  # Me
        8: 1.0,  # You
        9: 1.0,  # It
        10: 1.0, # Feel
        11: 1.0, # Happy
        12: 1.0, # Hungry
        13: 1.0, # Eat
        14: 1.0, # Bread
        15: 1.0, # Chocolate
        16: 1.0, # Tired
    }
    # Validate class weights
    print("Class weights:", class_weights)
    # Ensure the weights cover all classes
    assert len(class_weights) == number_of_classes

    if facemesh_included:
            coeficient = 1
            number_of_keypoints = 1662
    else:
            number_of_keypoints = 258
            coeficient = 1

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, number_of_keypoints), kernel_regularizer=l2(0.01)))
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