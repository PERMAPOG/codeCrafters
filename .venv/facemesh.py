import cv2
import itertools
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def facemeshapp():
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