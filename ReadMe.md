# Sign Language Detector using TensorFlow (Deep Learning) - <br/>Final Project for WBS Coding School 

<img src="https://github.com/Chrille91/Sign-Language-Detector/assets/49496538/35b5e37a-3569-4ff1-b15b-ba33e9dcad44.type" width="400" height="400">

## Presentation
Please find our presentation as PDF without videos here: [Vocal Hands Sign Language Detector Presentation](https://github.com/user-attachments/files/17902056/Vocal_Hands_public_small.pdf)  
I'd love to give you a full presentation incl. videos of our model in action, so don't hesitate to contact me here or on LinkedIn. ^_^ 

## Idea & Scope
We want to support people who use sign language as their main source of communication.  

The idea of our implementation is to address the current problems faced by existing AI projects dealing with sign language recognition. We are trying to provide smoother communication between people with hearing loss and those who do not know sign language.  

## Strategy

We chose American Sign Language (ASL) as our standard sign language because it offers more reliable data and a larger user base. _(Yes, there are many international sign languages, which is another barrier to barrier-free communication)._  

Based on a [tutorial by Nicholas Renotte](https://www.youtube.com/watch?v=doDUihpj6ro), we selected 15 from over 10,000 signs. Then, we recorded our own videos of us signing ASL, since existing sign language datasets are not standardized. Last but not least, we sixfolded the size of our video dataset through data augmentation. This technique employs image manipulations such as rotations, distortions, offsettings, and more to reduce the possibility of overfitting our model as well as boosting the stability of sign detection due to natural signing differences between signing people.  

To detect movements, we implement a pre-trained holistic landmark detection model from the [MediaPipe library](https://ai.google.dev/edge/mediapipe/solutions/guide). This model specializes on landmarking keypoints of the human body for gesture and body language recognition. Then, we set up a neural network using TensorFlow and Long Short-Term Memory (LSTM) layers which are suited for this type of dynamic movement detection and ressources available. After fine-tuning the model to our recorded data, keypoints of signs are being detected with a webcam and finally translated onto the user's webcam feed. As a bonus, we added the model's certainty as animated bars. 

## Features
- Live-translation of dynamic hand gestures (signs)
- Captioning of multiple successive signs 
- Certainty of detected sign
- Visible keypoint detection mask

## Limitations
The keypoint detection fails to discriminate between 2D and 3D movements, e.g. when hands are crossing during signing. 

## Future Goals
- Implement [new MediaPipe Gesture Recognizer model](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer), i.e. the successor of our MP holistic landmarks detection model
- Translation of national sign languages using subtitles
- Translating speech to hand signs
- Speech-captioning
- Lip-reading
- Body language detection
- ...

  
