import cv2
import itertools
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

#Face Swap Function for Live video
def face_swap_vid(image):
    app = FaceAnalysis(name="buffalo_l") #face detection model provided by insigthface
    app.prepare(ctx_id=0, det_size=(640,640))

    #pass in an image
    img = cv2.imread('will.png')
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

    return target

#Void function, call it and it will swap Will Smiths face on a picture of mine 
def face_swap_pic():
    app = FaceAnalysis(name="buffalo_l") #face detection model
    app.prepare(ctx_id=0, det_size=(640,640))

    #pass in an image
    img = cv2.imread('will.png')
    will_faces = app.get(img) #detect face in image
    will_face = will_faces[0]

    #Target image that wants to be altered
    target = cv2.imread('byron.jpg')
    target_faces = app.get(target)
    target_face = target_faces[0]

    #The code below is to test if it is detected the face
    #uncomment to test:
    #bbox = target_face['bbox']
    #bbox = [int(b) for b in bbox]
    #plt.imshow(target[bbox[1]:bbox[3],bbox[0]:bbox[2],::-1])
    #plt.show()

    res = target.copy() #make a copy of target photo to not alter original

    #Face swapping
    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False,download_zip=False)

    res = swapper.get(res, target_face, will_face, paste_back=True)

    #Used to display the image
    fig, ax = plt.subplots()
    ax.imshow(res[:,:,::-1])
    ax.axis('off')
    plt.show()
    


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




#------------------------------------
#Here is the way I have my webcam loop. I modified it to be able to call the functions. I also turned the facemesh into a function:
# This function applies the facemesh on the video

#def apply_face_mesh_filter(image):
    # Apply face mesh using MediaPipe
   # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # result = face_mesh.FaceMesh(refine_landmarks=True).process(img_rgb)

    ## Draw annotations on the image
    #if result.multi_face_landmarks:
       # for face_landmarks in result.multi_face_landmarks:
            # Drawing the tesselation on image
        #    mp_drawing.draw_landmarks(
         #       image=image,
          #      landmark_list=face_landmarks,
           #     connections=face_mesh.FACEMESH_TESSELATION,
            #    landmark_drawing_spec=None,
             #   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            #)

            # Drawing the contours on image
            #mp_drawing.draw_landmarks(
             #   image=image,
              #  landmark_list=face_landmarks,
               # connections=face_mesh.FACEMESH_CONTOURS,
                #landmark_drawing_spec=None,
                #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            #)

            # Drawing the Irises on image
            #mp_drawing.draw_landmarks(
             #   image=image,
              #  landmark_list=face_landmarks,
               # connections=face_mesh.FACEMESH_IRISES,
                #landmark_drawing_spec=None,
                #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            #)

    #return image




# Start webcam
#webcam = cv2.VideoCapture(0)

#while webcam.isOpened():
    # reads frame by frame, if frame is successfuly read, the returns true
 #   success, img = webcam.read()
  #  if success:
   #     img = face_swap(img)
        #img = apply_face_mesh_filter(img)
        
    #    cv2.imshow("Webcam Face Mesh", img)
     #   key = cv2.waitKey(20) & 0xFF
      #  if key == ord("q"):  # Press 'q' to exit
       #     break
        #elif key == ord("p"):  # Press 'p' to pause frame
         #   while cv2.waitKey(1) & 0xFF != ord("c"):  # Press 'c' to continue
          #      pass

