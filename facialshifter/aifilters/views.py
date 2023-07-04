from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.conf import settings
from django.urls import reverse
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import mediapipe as mp
from .webcam_filter.filter_call import cowboy_filter, load_overlay_images
from django.core.files.storage import FileSystemStorage
import os

# Create your views here.
def home(request):
    return render(request, 'index.html')

def cowboy_filter_view(request):
    return render(request, 'cowboy.html')

def apply_filter_view(request):
    # load filter overlay images
    overlay = load_overlay_images('COWBOY_FILTER')  # Change the filter name if needed

    # Mediapipe FaceDetection and FaceMesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    # Turn on the webcam
    cap = cv2.VideoCapture(0)

    # Sets up Mediapipe pre-trained models for face detection and face landmark detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
            mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            _, frame = cap.read()
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            result_face_detections = face_detection.process(frame_rgb)

            # if detections are present
            if result_face_detections.detections:
                for detection in result_face_detections.detections:
                    # Retrieves bounding box coordinates of detected face.
                    bbox = detection.location_data.relative_bounding_box

                    # Convert relative bounding box coordinates into absolute coordinates by scaling with width and height of frame.
                    # Necessary to draw the overlay images(hat and mustache) accurately onto the face.
                    frame_h, frame_w, _ = frame.shape
                    x = int(bbox.xmin * frame_w)
                    y = int(bbox.ymin * frame_h)
                    w = int(bbox.width * frame_w)
                    h = int(bbox.height * frame_h)
                    coordinates = (x, y, w, h)
                    frame_shape = (frame_h, frame_w)

                    # Convert frame to BGR to prepare for face landmark detection
                    frame_landmarks = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Optimizes face landmark detection process
                    frame_landmarks.flags.writeable = False
                    # Detect all face landmarks (features) - top of the head, lips, nose, eye, jaw, etc.
                    results_face_features = face_mesh.process(frame_landmarks)

                    # Detect all faces in the frame (one or more)
                    if results_face_features.multi_face_landmarks:
                        # Select the first detected face
                        face_features = results_face_features.multi_face_landmarks[0]

                        # Filter function call
                        frame = cowboy_filter(frame, overlay, face_features, frame_shape, coordinates)
                        
            # Display the frame with overlays
            cv2.imshow('Face Features Detection and Filters', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    return JsonResponse({'status': 'success'})

def result_view(request, img_url):
    return render(request, 'result.html', {'processed_image': img_url})

def face_mesh_page_view(request):
    return render(request, 'facemesh.html')

def face_mesh_view(request):
    # Initialize face_mesh and mp_drawing
    face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if request.method == 'POST':
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        image_url = fs.url(filename)

        # load the image
        img = cv2.imread(image_url[1:])
        
        # Apply face mesh using MediaPipe
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

        out_image_name = "output_" + image.name
        output_path = os.path.join(settings.MEDIA_ROOT, out_image_name)
        cv2.imwrite(output_path, img)
        img_url = fs.url(out_image_name)[1:]  # Remove leading slash from the URL

        # Generate the URL for the result view
        result_url = reverse('result', kwargs={'img_url': img_url})
        return redirect(result_url)
    
    return render(request, 'upload.html')