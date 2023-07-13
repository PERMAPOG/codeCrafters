"""
facemesh.py

Description: This code captures frames from the webcam and performs live face swapping using the insightface library.
It also does the face swap on static images.
It also includes a face mesh detection application using MediaPipe.

Author: Byron Munguia Najera
Version: 1.0
Date: 7/11/2023

Modules:
- cv2: OpenCV for image and video processing.
- itertools: Provides various functions for efficient iteration.
- numpy: NumPy for numerical operations.
- mediapipe: MediaPipe for face mesh detection.
- matplotlib.pyplot: Matplotlib for displaying images.
- insightface: InsightFace for face analysis and face swapping.

Functions:
- captureFrame(): Captures frames from the webcam and displays a live face-swapped video.
- face_swap_vid(image): Performs face swapping on a video frame using the insightface library.
- face_swap_pic(): Performs face swapping on a static image using the insightface library.
- facemeshapp(): Runs a face mesh detection application using MediaPipe.

"""

import cv2
import itertools
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

def captureFrame():

    """
    Captures frames from the webcam and displays a live face-swapped video.

    Uses the OpenCV library to access the webcam and perform live face swapping using the insightface library.

    Press 'q' to quit the video stream.

    Args:
        None

    Returns:
        None
    """
    # Turn on the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Check if frame was successfully read
        if not ret:
            print("Failed to grab frame")
            break

        swapped_frame = face_swap_vid(frame)
        
        # Display the resulting frame
        cv2.imshow('Face Swapped Video', swapped_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    return

#Face Swap Function for Live video
def face_swap_vid(image):
    """
    Performs face swapping on a video frame.

    Uses the insightface library and a pre-trained face detection model to detect and swap faces in a video frame.

    Args:
        image: The video frame to perform face swapping on.

    Returns:
        The video frame with swapped faces.
    """
    app = FaceAnalysis(name="buffalo_l") #face detection model provided by insigthface
    app.prepare(ctx_id=0, det_size=(640,640))

    #pass in an image
    img = cv2.imread('../Static/images/will.png')
    will_faces = app.get(img) #detect face in image
    will_face = will_faces[0]

    # Converts the color of the image from BGR to RGB format.
    #OpenCV wont be able to work with it otherwise
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target = img_rgb
    target_faces = app.get(target)
    target_face = target_faces[0]
    
    #Face swapping
    #inswapper_128.onnx trained model downloaded from Insightface github tutorial
    #here is the url: https://github.com/deepinsight/insightface/tree/master/examples/in_swapper
    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False,download_zip=False)

    target = swapper.get(target, target_face, will_face, paste_back=True)
    
    # Convert the color of the image back from RGB to BGR format.
    img_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

    return img_bgr

#Void function, call it and it will swap Will Smiths face on a picture of mine 
def face_swap_pic():

    """
    Performs face swapping on a static image.

    Uses the insightface library and a pre-trained face detection model to detect and swap faces in a static image.

    Displays the resulting image after face swapping.

    Args:
        None

    Returns:
        None
    """
    app = FaceAnalysis(name="buffalo_l") #face detection model
    app.prepare(ctx_id=0, det_size=(640,640))

    #pass in an image
    img = cv2.imread('../Static/images/will.png')
    will_faces = app.get(img) #detect face in image
    will_face = will_faces[0]

    #Target image that wants to be altered
    target = cv2.imread('../Static/images/byron.jpeg')
    target_faces = app.get(target)
    target_face = target_faces[0]



    res = target.copy() #make a copy of target photo to not alter original

    #Face swapping
    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False,download_zip=False)

    res = swapper.get(res, target_face, will_face, paste_back=True)

    #Used to display the image
    fig, ax = plt.subplots()
    ax.imshow(res[:,:,::-1])
    ax.axis('off')
    plt.show()

    return

def facemeshapp():

    """
    Runs a face mesh detection application using MediaPipe.

    Opens the webcam and applies face mesh detection using the MediaPipe library.
    Draws annotations on the video frame, including tesselation, contours, and irises.

    Press 'q' to quit the application.

    Returns:
        None
    """
    # mediapipe face mesh
    face_mesh = mp.solutions.face_mesh

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    webcam = cv2.VideoCapture(0)

    while webcam.isOpened():
        success, img = webcam.read()

        # Applying face mesh using MediaPipe
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.FaceMesh(refine_landmarks=True).process(img)

        # Draw annotations on the image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Drawing the tesselation on image
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # Drawing the contours on image
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                # Drawing the Irises on image
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        cv2.imshow("Face Mesh", img)
        if cv2.getWindowProperty('Face Mesh', cv2.WND_PROP_VISIBLE) < 1: 
            break
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
