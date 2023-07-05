import cv2
import numpy as np
import mediapipe as mp

# mediapipe face mesh 
face_mesh = mp.solutions.face_mesh

# read in sunglasses & convert sunglasses from BGR to RGBA
sunglasses = cv2.imread('.venv/Static/images/bossMode.png', cv2.IMREAD_UNCHANGED)
sunglasses = cv2.cvtColor(sunglasses, cv2.COLOR_BGRA2RGBA)

# parameters for face mesher
face_mesh_images = face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.7)

# list of image paths
image_files = ['.venv/Static/images/sample7.jpg', '.venv/Static/images/sample2.jpg', '.venv/Static/images/sample8.jpg']

# Lists to store original and filtered images
original_images = []
filtered_images = []

for image_file in image_files:

    # read in image & convert BGR to RGB
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Store a copy of the original image
    original_images.append(img.copy())

    # face mesh desired image
    face_mesh_results = face_mesh_images.process(img)

    # if face mesh found in image
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:

            left_eye_corner = (int(face_landmarks.landmark[226].x * img.shape[1]), int(face_landmarks.landmark[226].y * img.shape[0]))
            right_eye_corner = (int(face_landmarks.landmark[446].x * img.shape[1]), int(face_landmarks.landmark[446].y * img.shape[0]))

            # calculates the desired width of the sunglasses filter which is three times the distance between the eyes
            sunglasses_width = abs(right_eye_corner[0] - left_eye_corner[0]) # Calculate width by difference in X-coordinates (0, 1 is y cord)
            sunglasses_width = int(3.3 * sunglasses_width)
            ratio = sunglasses_width / sunglasses.shape[1] 

            # Resize the sunglasses to fit the face width
            if ratio > 0:
                new_sunglasses = cv2.resize(sunglasses, (0,0), fx=ratio, fy=ratio)

                mid_point = ((right_eye_corner[0] + left_eye_corner[0]) // 2, (right_eye_corner[1] + left_eye_corner[1]) // 2)
                start_point = (mid_point[0] - new_sunglasses.shape[1] // 2, mid_point[1] - new_sunglasses.shape[0] // 2)
                start_point = (max(0, start_point[0]), max(0, start_point[1]))

                img_region = img[start_point[1]:start_point[1] + new_sunglasses.shape[0], start_point[0]:start_point[0] + new_sunglasses.shape[1]]

                alpha_sunglasses = new_sunglasses[:,:,3] / 255.0
                alpha_img = 1.0 - alpha_sunglasses
                img_region[:,:,:3] = (alpha_sunglasses[:,:,np.newaxis] * new_sunglasses[:,:,:3] + alpha_img[:,:,np.newaxis] * img_region[:,:,:3])


    # Store the filtered image
    filtered_images.append(img)

# display the original and filtered images
for i in range(len(original_images)):
    original = original_images[i]
    filtered = filtered_images[i]

    #display image with keyboard wait
    cv2.imshow('Original', cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0) 

    cv2.imshow('Filtered', cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(0) 

    if key == ord('q'):  # Quits if q is pressed
        break

cv2.destroyAllWindows()
