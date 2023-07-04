import cv2
import numpy as np
def load_overlay_images(filter):
    if filter == 'COWBOY_FILTER':
        # Load the hat and mustache images
        hat_image = cv2.imread('./Static/images/hat.png', -1)
        mustache_image = cv2.imread('./Static/images/mustache.png', -1)
        return hat_image, mustache_image
    elif filter == 'SUNGLASSES_FILTER':
        # Load the boss mode image
        img = cv2.imread('./Static/images/bossMode.png', -1)
        return img
    elif filter == '2':
        # load overlay
        pass
    else:
        # else
        return

def cowboy_filter(frame, overlay, face_landmarks, frame_shape, coordinates):
    hat_image, mustache_image = overlay
    frameh, framew = frame_shape
    x, y, w, h = coordinates

    left_eye_x = face_landmarks.landmark[33].x * framew
    right_eye_x = face_landmarks.landmark[263].x * framew
    eye_distance = abs(left_eye_x - right_eye_x)

    # Get coordinates for top of the head (hat)
    # Adjust as needed
    top_head_x = int(face_landmarks.landmark[10].x * framew)
    top_head_y = int(face_landmarks.landmark[10].y * frameh)

    # Get coordinates for the upper lip (mustache)
    # Adjust as needed
    upper_lip_x = int(face_landmarks.landmark[13].x * framew)
    upper_lip_y = int(face_landmarks.landmark[13].y * frameh)


    # Overlay the hat on top of the head
    hat_height = int(eye_distance * 2.5) # adjust this factor to get the desired hat size
    hat_width = int(hat_height * hat_image.shape[1] / hat_image.shape[0])


    # Ensure the hat is within the frame
    hat_x = top_head_x - int(hat_width / 2)
    hat_y = top_head_y - hat_height
    hat_y = max(hat_y, 0)
    hat_x = max(hat_x, 0)
    hat_y_end = min(hat_y + hat_height, frameh)
    hat_x_end = min(hat_x + hat_width, framew)

    # resize hat overlay to match calculated dimensions
    hat_resized = cv2.resize(hat_image, (hat_x_end-hat_x, hat_y_end-hat_y)) # note the updated size

    # Apply alpha blending for transparent overlay (needed for png transparency in imshow())
    alpha_hat = hat_resized[0:hat_y_end-hat_y, 0:hat_x_end-hat_x, 3] / 255.0

    # Overlay the hat on the frame using alpha blending
    # Overlay moving out of frame causes Exception error.
    try:
        for c in range(0, 3):
            frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width, c] = alpha_hat * hat_resized[:, :, c] + (
                    1 - alpha_hat) * frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width, c]
    except Exception as e:
        print("An exception occurred:", str(e))

    # Overlay the mustache on the upper lip
    # mustache SIZE(adjust as need)
    mustache_height = int(h * 0.95)
    mustache_width = int(mustache_height * mustache_image.shape[1] / mustache_image.shape[0])   # proportional to mustache_height

    # mustache_x and mustache_y are top left coordiantes of mustache overlay
    mustache_x = upper_lip_x - int(mustache_width / 2)      # Centers mustache horizonally on the upper lip
    mustache_y = upper_lip_y - int(mustache_height / 2)     # centers mustache vertically on upper lip

    # Ensure the mustache is within the frame
    mustache_y = max(mustache_y, 0)
    mustache_x = max(mustache_x, 0)
    mustache_y_end = min(mustache_y + mustache_height, frameh)
    mustache_x_end = min(mustache_x + mustache_width, framew)

    # resize mustache overlay to match calculated dimensions
    mustache_resized = cv2.resize(mustache_image, (mustache_x_end-mustache_x, mustache_y_end-mustache_y)) # note the updated size

    # Apply alpha blending for transparent overlay (needed for png transparency in imshow())
    alpha_mustache = mustache_resized[0:mustache_y_end-mustache_y, 0:mustache_x_end-mustache_x, 3] / 255.0

    # Overlay the hat on the frame using alpha blending
    # Overlay moving out of frame causes Exception error.
    try:
        for c in range(0, 3):
            frame[mustache_y:mustache_y + mustache_height, mustache_x:mustache_x + mustache_width,
            c] = alpha_mustache * mustache_resized[:, :, c] + (
                    1 - alpha_mustache) * frame[mustache_y:mustache_y + mustache_height,
                                          mustache_x:mustache_x + mustache_width, c]
    except Exception as e:
        print("An exception occurred:", str(e))

    return frame

def bossModeFilter(frame, overlay, face_landmarks, frame_shape):
    img = overlay
    frameh, framew = frame_shape
    left_eye_x = face_landmarks.landmark[33].x * framew
    right_eye_x = face_landmarks.landmark[263].x * framew
    eye_distance = abs(left_eye_x - right_eye_x)

    # Get coordinates for top of the head
    top_head_x = int(face_landmarks.landmark[10].x * framew)
    top_head_y = int(face_landmarks.landmark[10].y * frameh)


    # Overlay the image on top of the head
    img_height = int(eye_distance * 6) # adjust this factor to get the desired hat size
    img_width = int(img_height * img.shape[1] / img.shape[0])

    # Ensure the image is within the frame
    img_x = top_head_x - int(img_width / 2)
    img_y = top_head_y - img_height
    img_y = max(img_y, 0)
    img_x = max(img_x, 0)
    img_y_end = min(img_y + img_height, frameh)
    img_x_end = min(img_x + img_width, framew)

    # resize hat overlay to match calculated dimensions
    img_resized = cv2.resize(img, (img_x_end-img_x, img_y_end-img_y)) # note the updated size

    # Apply alpha blending for transparent overlay (needed for png transparency in imshow())
    alpha_img = img_resized[0:img_y_end-img_y, 0:img_x_end-img_x, 3] / 255.0

    # Overlay the hat on the frame using alpha blending
    # Overlay moving out of frame causes Exception error.
    try:
        for c in range(0, 3):
            frame[img_y:img_y + img_height, img_x:img_x + img_width, c] = alpha_img * img_resized[:, :, c] + (
                    1 - alpha_img) * frame[img_y:img_y + img_height, img_x:img_x + img_width, c]
    except Exception as e:
        print("An exception occurred:", str(e))

    return frame