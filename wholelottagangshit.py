import cv2
import itertools
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# mediapipe drawing utlilites 
drawing_tools = mp.solutions.drawing_utils

# mediapipe face mesh 
face_mesh = mp.solutions.face_mesh

drawing_styles = mp.solutions.drawing_styles

img = cv2.imread('static/images/sample.jpg')
sunglasses = cv2.imread('static/images/sunglasses.jpg')

face_mesh_images = face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)

# formatt to RGB
face_mesh_results = face_mesh_images.process(img[:,:,::-1])

LEFT_EYE_INDEXES = list(set(itertools.chain(*face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*face_mesh.FACEMESH_RIGHT_EYE)))

img_copy = img[:,:,::-1].copy()

if face_mesh_results.multi_face_landmarks:
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        # Get the coordinates of the corners of the sunglasses
        left_eye_corner = (int(face_landmarks.landmark[2].x * img.shape[1]), int(face_landmarks.landmark[2].y * img.shape[0]))
        right_eye_corner = (int(face_landmarks.landmark[338].x * img.shape[1]), int(face_landmarks.landmark[338].y * img.shape[0]))

        # Calculate sunglasses width and the ratio
        sunglasses_width = left_eye_corner[0]
          # Increase the width by 20%
        sunglasses_width = int(1.2 * sunglasses_width)
        ratio = sunglasses_width / sunglasses.shape[1] if sunglasses.shape[1] != 0 else 0

        # Resize the sunglasses to fit the face width
        if ratio > 0:
            new_sunglasses = cv2.resize(sunglasses, (0,0), fx=ratio, fy=ratio)

            # Calculate the mid-point between eyes
            mid_point = ((right_eye_corner[0] + left_eye_corner[0]) // 2, (right_eye_corner[1] + left_eye_corner[1]) // 2)
            # Adjust starting point to make sunglasses center align with eyes mid-point
            start_point = (mid_point[0] - new_sunglasses.shape[1] // 2, mid_point[1] - new_sunglasses.shape[0] // 2)
            # Ensure the starting point is not out of image boundary
            start_point = (max(0, start_point[0]), max(0, start_point[1]))

            # Find the region in the image where we want to add the sunglasses
            sunglasses_region = img_copy[start_point[1]:start_point[1] + new_sunglasses.shape[0], start_point[0]:start_point[0] + new_sunglasses.shape[1]]

            # Use bitwise operations to merge the sunglasses onto the image
            img_copy[start_point[1]:start_point[1] + new_sunglasses.shape[0], start_point[0]:start_point[0] + new_sunglasses.shape[1]] = cv2.bitwise_and(sunglasses_region, new_sunglasses)

plt.figure(figsize = [10, 10])
plt.imshow(img_copy)
plt.axis('off')
plt.show()
