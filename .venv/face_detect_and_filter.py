import cv2
import mediapipe as mp
from filter_call import cowboy_filter, load_overlay_images, bossModeFilter, police_filter, pirate_filter

def face_detect_and_filter(filter):
    # load filter overlay images
    overlay = load_overlay_images(filter)

    # Mediapipe FaceDetection and FaceMesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    # Turn on the webcam
    cap = cv2.VideoCapture(0)

    # Sets up Mediapipe pre-trained models for face detection and fae landmark detection
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
                    # Necessary to draw the overlay images(hat and mustache) accurately onto face.
                    frameh, framew, _ = frame.shape
                    x = int(bbox.xmin * framew)
                    y = int(bbox.ymin * frameh)
                    w = int(bbox.width * framew)
                    h = int(bbox.height * frameh)
                    coordinates = (x, y, w, h)
                    frame_shape = (frameh, framew)

                    # Convert frame to BGR to prepare for face landmark detection
                    frame_landmarks = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Optimizes face landmark detection process
                    frame_landmarks.flags.writeable = False
                    # Detect all face landmarks(features) (top of head, lips, nose, eye, jaw, etc)
                    results_face_features = face_mesh.process(frame_landmarks)

                    # Detects all faces in frame (one or more)
                    if results_face_features.multi_face_landmarks:

                        # Selects the first detected face
                        face_features = results_face_features.multi_face_landmarks[0]


                        # filter function call!!!!!!!!!!!!!!
                        if filter == 'COWBOY_FILTER':
                            frame = cowboy_filter(frame, overlay, face_features, frame_shape, coordinates)
                        elif filter == 'BOSSMODE_FILTER':
                            # function call here
                            frame = bossModeFilter(frame, overlay, face_features, frame_shape)
                        elif filter == 'POLICE_FILTER':
                            # function call here
                            frame = police_filter(frame, overlay, face_features, frame_shape, coordinates)
                        elif filter == 'PIRATE_FILTER':
                            frame = pirate_filter(frame, overlay, face_features, frame_shape, coordinates)
                        else:
                            # else
                            pass

            # Display the frame with overlays
            cv2.imshow(filter, frame)
            
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()



# TEST
#face_detect_and_filter('COWBOY_FILTER')
#face_detect_and_filter('POLICE_FILTER')
#face_detect_and_filter('PIRATE_FILTER')
#face_detect_and_filter('SUNGLASSES_FILTER')