import cv2

def load_overlay_images(filter):
    if filter == 'COWBOY_FILTER':
        # Load the hat and mustache images
        hat_image = cv2.imread('static/filters/cowboy/hat.png', -1)
        mustache_image = cv2.imread('static/filters/cowboy/mustache.png', -1)
        return hat_image, mustache_image
    elif filter == '1':
        # function call here
        pass
    elif filter == '2':
        # function call here
        pass
    else:
        # else
        return



def cowboy_filter(frame, overlay, face_landmarks, frame_shape, coordinates):
    hat_image, mustache_image = overlay
    frameh, framew = frame_shape
    x, y, w, h = coordinates

    # Get coordinates for top of the head (hat)
    # Adjust as needed
    top_head_x = int(face_landmarks.landmark[10].x * framew)
    top_head_y = int(face_landmarks.landmark[10].y * frameh) + 80

    # Get coordinates for the upper lip (mustache)
    # Adjust as needed
    upper_lip_x = int(face_landmarks.landmark[13].x * framew)
    upper_lip_y = int(face_landmarks.landmark[13].y * frameh)



    # Overlay the hat on top of the head
    # hat SIZE(adjust as need)
    hat_height = int(h * 1.9)
    hat_width = int(hat_height * hat_image.shape[1] / hat_image.shape[0])   # proportional to hat_height

    # hat_x and hat_y are top left corner coordinates of hat overlay
    hat_x = top_head_x - int(hat_width / 2)     # Ensures the hat is centered horizontally on head
    hat_y = top_head_y - hat_height     # Ensures the hat is above the top of the head

    # resize hat overlay to match calculated dimensions
    hat_resized = cv2.resize(hat_image, (hat_width, hat_height))

    # Apply alpha blending for transparent overlay (needed for png transparency in imshow())
    alpha_hat = hat_resized[:, :, 3] / 255.0
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

    # resize mustache overlay to match calculated dimensions
    mustache_resized = cv2.resize(mustache_image, (mustache_width, mustache_height))

    # Apply alpha blending for transparent overlay (needed for png transparency in imshow())
    alpha_mustache = mustache_resized[:, :, 3] / 255.0
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