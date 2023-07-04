import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# mediapipe drawing utilities 
#drawing_tools = mp.solutions.drawing_utils

# mediapipe face mesh 
face_mesh = mp.solutions.face_mesh

# mediapipe drawing styles
drawing_styles = mp.solutions.drawing_styles

# read in image 
img = cv2.imread('static/images/sample5.jpg', cv2.IMREAD_UNCHANGED)
# convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# read in sunglasses
sunglasses = cv2.imread('static/images/bossMode.png', cv2.IMREAD_UNCHANGED)
# convert sunglasses from BGR to RGBA
sunglasses = cv2.cvtColor(sunglasses, cv2.COLOR_BGRA2RGBA)

# parameters for face mesher
face_mesh_images = face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.7)

# face mesh desired image
face_mesh_results = face_mesh_images.process(img)

# if face mesh found in image
if face_mesh_results.multi_face_landmarks:
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        
        left_eye_corner = (int(face_landmarks.landmark[46].x * img.shape[1]), int(face_landmarks.landmark[446].y * img.shape[0]))
        right_eye_corner = (int(face_landmarks.landmark[276].x * img.shape[1]), int(face_landmarks.landmark[113].y * img.shape[0]))

        # calculate sunglasses width and the ratio
        sunglasses_width = abs(right_eye_corner[0] - left_eye_corner[0]) # Calculate width by difference in X-coordinates
        sunglasses_width = int(3 * sunglasses_width)
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

            # image reagion where we want to add the sunglasses
            img_region = img[start_point[1]:start_point[1] + new_sunglasses.shape[0], start_point[0]:start_point[0] + new_sunglasses.shape[1]]

            # Merge the sunglasses onto the image using alpha blending
            #0 indicates fully transparent and 255 indicates fully opaque.
            alpha_sunglasses = new_sunglasses[:,:,3] / 255.0
            alpha_img = 1.0 - alpha_sunglasses
            for c in range(0, 3):
                img_region[:,:,c] = (alpha_sunglasses * new_sunglasses[:,:,c] +
                                    alpha_img * img_region[:,:,c])


plt.figure(figsize = [10, 10])
plt.imshow(img)
plt.axis('off')
plt.show()
