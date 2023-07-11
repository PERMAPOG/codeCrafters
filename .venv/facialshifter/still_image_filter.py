"""
still_image_filter.py

Description: This code applies a filter to images, placing the filter onto faces detected in the images.

Author: Christian Jaime
Version: 1.0
Date: 7/7/2023

Modules:
- cv2: OpenCV for image processing.
- numpy: NumPy for numerical operations.
- mediapipe: MediaPipe for face mesh detection.

Functions:
- read_image(image_file)
- apply_filter(img, face_mesh_images, filter)
- get_filter_params(face_landmarks, img_shape, filter_shape)
- overlay_filter(img, new_filter, left_eye_corner, right_eye_corner)
- display_images(original_images, filtered_images)
- bossModeImage() [main function]
"""

import cv2
import numpy as np
import mediapipe as mp


def read_image(image_file):
    """
    Reads an image from a file and converts it from BGR to RGB.

    Args:
        image_file: The path of the image file.
    
    Returns:
        The image in RGB.
    """
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def apply_filter(img, face_mesh_images, filter):
    """
    Applies a filter to a face in an image.

    Args:
        img: The image that the filter will be placed ontop of.
        face_mesh_images: The Mediapipe FaceMesh object.
        filter: The filter that will be applied.
    
    Returns:
        The image with the filter applied.
    """

    face_mesh_results = face_mesh_images.process(img)

    # if face mesh found in image
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:

            # calculates the desired width of the filter which is 3.3 times the distance between the eyes
            left_eye_corner, right_eye_corner, ratio = get_filter_params(face_landmarks, img.shape, filter.shape)

            # Resize the filter to fit the face width
            if ratio > 0:
                resized_filter = cv2.resize(filter, (0,0), fx=ratio, fy=ratio)
                img = overlay_filter(img, resized_filter, left_eye_corner, right_eye_corner)

    return img

def get_filter_params(face_landmarks, img_shape, filter_shape):
    """
    Calculates the parameters for applying filter.

    Args:
        face_landmarks: The detected face landmarks.
        img_shape: The shape of the image.
        filter_shape: The shape of the filter.
    
    Returns:
        A args of the left eye corner, right eye corner, filter width and ratio.
    """
    left_eye_corner = (int(face_landmarks.landmark[226].x * img_shape[1]), int(face_landmarks.landmark[226].y * img_shape[0]))
    right_eye_corner = (int(face_landmarks.landmark[446].x * img_shape[1]), int(face_landmarks.landmark[446].y * img_shape[0]))

    filter_width = abs(right_eye_corner[0] - left_eye_corner[0]) # Calculate width by difference in X-coordinates
    filter_width = int(3.3 * filter_width)
    ratio = filter_width / filter_shape[1]

    return left_eye_corner, right_eye_corner, ratio

def overlay_filter(img, new_filter, left_eye_corner, right_eye_corner):
    """
    Overlays the filter onto the image.

    Args:
        img: The image that the filter will be applied to.
        new_filter: The resized filter.
        left_eye_corner: The coordinates of the left eye corner.
        right_eye_corner: The coordinates of the right eye corner.
    
    Returns:
        The image with the filter overlayed.
    """

    
    mid_point = ((right_eye_corner[0] + left_eye_corner[0]) // 2, (right_eye_corner[1] + left_eye_corner[1]) // 2) # // in Python performs integer floor division nessary for pixel placement
    start_point = (mid_point[0] - new_filter.shape[1] // 2, mid_point[1] - new_filter.shape[0] // 2)
    
    # ensures that the starting point of the filter is within the image boundaries
    start_point = (max(0, start_point[0]), max(0, start_point[1]))
    img_region = img[start_point[1]:start_point[1] + new_filter.shape[0], start_point[0]:start_point[0] + new_filter.shape[1]]


    #  the transparency level of a pixel (0-255)
    alpha_filter = new_filter[:,:,3] / 255.0
    alpha_img = 1.0 - alpha_filter

    # applies the filter to image region
    img_region[:,:,:3] = (alpha_filter[:,:,np.newaxis] * new_filter[:,:,:3] + alpha_img[:,:,np.newaxis] * img_region[:,:,:3])
    
    return img

def display_images(original_images, filtered_images):
    """
    Displays the original and filtered images.

    Args:
        original_images: A list of original images.
        filtered_images: A list of filtered images.
    """
    for i in range(len(original_images)):
        original = original_images[i]
        filtered = filtered_images[i]

        cv2.imshow('Original', cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0) 

        cv2.imshow('Filtered', cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0) 

        if key == ord('q'):  # Quits if q is pressed
            break

    cv2.destroyAllWindows()

def bossModeImage():
    """
    The main function that runs the script.
    """
    face_mesh = mp.solutions.face_mesh
    imported_filter = cv2.imread('.venv/Static/images/bossMode.png', cv2.IMREAD_UNCHANGED)
    imported_filter = cv2.cvtColor(imported_filter, cv2.COLOR_BGRA2RGBA)

    face_mesh_images = face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.7)

    image_files = ['.venv/Static/images/sample7.jpg', '.venv/Static/images/sample2.jpg', '.venv/Static/images/sample8.jpg']

    original_images = []
    filtered_images = []

    for image_file in image_files:
        img = read_image(image_file)
        original_images.append(img.copy())
        img = apply_filter(img, face_mesh_images, imported_filter)
        filtered_images.append(img)

    display_images(original_images, filtered_images)

