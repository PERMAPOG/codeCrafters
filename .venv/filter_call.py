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
    elif filter == 'PIRATE_FILTER':
        # load overlay
        pirate_hat = cv2.imread('./Static/images/pirate_hat.png', -1)
        pirate_eyepatch = cv2.imread('./Static/images/pirate_eyepatch.png', -1)
        pirate_beard = cv2.imread('./Static/images/pirate_beard.png', -1)
        return pirate_hat, pirate_eyepatch, pirate_beard
    elif filter == 'POLICE_FILTER':
        # load overlay
        police_hat = cv2.imread('./Static/images/police_hat.png', -1)
        police_glasses = cv2.imread('./Static/images/police_glasses.png', -1)
        police_mustache = cv2.imread('./Static/images/police_mustache.png', -1)
        return police_hat, police_glasses, police_mustache
    else:
        # else
        return




def cowboy_filter(frame, overlay, face_landmarks, frame_shape, coordinates):
    hat_image, mustache_image = overlay
    frameh, framew = frame_shape
    x, y, w, h = coordinates

    # Get coordinates for left eye and right eye and then calculate distance between them
    left_eye_x = face_landmarks.landmark[33].x * framew
    right_eye_x = face_landmarks.landmark[263].x * framew
    eye_distance = abs(left_eye_x - right_eye_x)

    # Get coordinates for top of the head (hat)
    # Adjust as needed
    top_head_x = int(face_landmarks.landmark[10].x * framew)
    top_head_y = int(face_landmarks.landmark[10].y * frameh) + 90

    # Get coordinates for the upper lip (mustache)
    # Adjust as needed
    upper_lip_x = int(face_landmarks.landmark[13].x * framew)
    upper_lip_y = int(face_landmarks.landmark[13].y * frameh)


    # Overlay the hat on top of the head
    hat_height = int(eye_distance * 3.65) # adjust this factor to get the desired hat size
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

    # Get coordinates for left eye and right eye and then calculate distance between them
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



def police_filter(frame, overlay, face_landmarks, frame_shape, coordinates):
    police_hat, police_glasses, police_mustache = overlay
    frameh, framew = frame_shape
    x, y, w, h = coordinates

    # Get coordinates for left eye and right eye and then calculate distance between them
    left_eye_x = face_landmarks.landmark[33].x * framew
    right_eye_x = face_landmarks.landmark[263].x * framew
    eye_distance = abs(left_eye_x - right_eye_x)

    # Get coordinates between left and right eye
    between_eyes_x = int((left_eye_x + right_eye_x) / 2)
    between_eyes_y = int(face_landmarks.landmark[33].y * frameh)  # Assuming both eyes have the same y-coordinate

    # Get coordinates for top of the head (hat)
    # Adjust as needed
    top_head_x = int(face_landmarks.landmark[10].x * framew)
    top_head_y = int(face_landmarks.landmark[10].y * frameh) + 90

    # Get coordinates for the upper lip (mustache)
    # Adjust as needed
    upper_lip_x = int(face_landmarks.landmark[13].x * framew)
    upper_lip_y = int(face_landmarks.landmark[13].y * frameh)


    # Overlay the hat on top of the head
    hat_height = int(eye_distance * 3) # adjust this factor to get the desired hat size
    hat_width = int(hat_height * police_hat.shape[1] / police_hat.shape[0])


    # Ensure the hat is within the frame
    hat_x = top_head_x - int(hat_width / 2)
    hat_y = top_head_y - hat_height
    hat_y = max(hat_y, 0)
    hat_x = max(hat_x, 0)
    hat_y_end = min(hat_y + hat_height, frameh)
    hat_x_end = min(hat_x + hat_width, framew)

    # resize hat overlay to match calculated dimensions
    hat_resized = cv2.resize(police_hat, (hat_x_end-hat_x, hat_y_end-hat_y)) # note the updated size

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



    # GLASSES
    # glasses size (adjust as needed)
    glasses_height = int(eye_distance * 2)
    glasses_width = int(glasses_height * police_glasses.shape[1] / police_glasses.shape[0])

    # Calculate the coordinates for the top-left corner of the glasses
    glasses_x = between_eyes_x - int(glasses_width / 2)
    glasses_y = between_eyes_y - int(glasses_height / 2)

    # Ensure the glasses are within the frame
    glasses_y = max(glasses_y, 0)
    glasses_x = max(glasses_x, 0)
    glasses_y_end = min(glasses_y + glasses_height, frameh)
    glasses_x_end = min(glasses_x + glasses_width, framew)

    # Resize the glasses overlay to match the calculated dimensions
    glasses_resized = cv2.resize(police_glasses, (glasses_x_end - glasses_x, glasses_y_end - glasses_y))

    # Apply alpha blending for transparent overlay
    alpha_glasses = glasses_resized[:, :, 3] / 255.0

    # Overlay the glasses on the frame using alpha blending
    try:
        for c in range(0, 3):
            frame[glasses_y:glasses_y_end, glasses_x:glasses_x_end, c] = alpha_glasses * glasses_resized[:, :, c] + (1 - alpha_glasses) * frame[
                                                                                            glasses_y:glasses_y_end,
                                                                                        glasses_x:glasses_x_end, c]
    except Exception as e:
        print("An exception occurred:", str(e))
    


    # Overlay the mustache on the upper lip
    # mustache SIZE(adjust as need)
    mustache_height = int(h * 0.95)
    mustache_width = int(mustache_height * police_mustache.shape[1] / police_mustache.shape[0])   # proportional to mustache_height

    # mustache_x and mustache_y are top left coordiantes of mustache overlay
    mustache_x = upper_lip_x - int(mustache_width / 2)      # Centers mustache horizonally on the upper lip
    mustache_y = upper_lip_y - int(mustache_height / 2)     # centers mustache vertically on upper lip

    # Ensure the mustache is within the frame
    mustache_y = max(mustache_y, 0)
    mustache_x = max(mustache_x, 0)
    mustache_y_end = min(mustache_y + mustache_height, frameh)
    mustache_x_end = min(mustache_x + mustache_width, framew)

    # resize mustache overlay to match calculated dimensions
    mustache_resized = cv2.resize(police_mustache, (mustache_x_end-mustache_x, mustache_y_end-mustache_y)) # note the updated size

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




def pirate_filter(frame, overlay, face_landmarks, frame_shape, coordinates):
    pirate_hat, pirate_eyepatch, pirate_beard = overlay
    frameh, framew = frame_shape
    x, y, w, h = coordinates

    # Get coordinates for left eye and right eye and then calculate distance between them
    left_eye_x = face_landmarks.landmark[33].x * framew
    right_eye_x = face_landmarks.landmark[263].x * framew
    eye_distance = abs(left_eye_x - right_eye_x)

    # Get coordinates between left and right eye
    between_eyes_x = int((left_eye_x + right_eye_x) / 2)
    between_eyes_y = int(face_landmarks.landmark[33].y * frameh) - 5 # Assuming both eyes have the same y-coordinate

    # Get coordinates for top of the head (hat)
    # Adjust as needed
    top_head_x = int(face_landmarks.landmark[10].x * framew)
    top_head_y = int(face_landmarks.landmark[10].y * frameh) + 60

    # Get coordinates for the upper lip (mustache)
    # Adjust as needed
    upper_lip_x = int(face_landmarks.landmark[13].x * framew)
    upper_lip_y = int(face_landmarks.landmark[13].y * frameh) + 30



    # EYEPATCH
    # eyepatch size (adjust as needed)
    eyepatch_height = int(eye_distance * 2.2)
    eyepatch_width = int(eyepatch_height * pirate_eyepatch.shape[1] / pirate_eyepatch.shape[0])

    # Calculate the coordinates for the top-left corner of the eyepatch
    eyepatch_x = between_eyes_x - int(eyepatch_width / 2)
    eyepatch_y = between_eyes_y - int(eyepatch_height / 2)

    # Ensure the eyepatch is within the frame
    eyepatch_y = max(eyepatch_y, 0)
    eyepatch_x = max(eyepatch_x, 0)
    eyepatch_y_end = min(eyepatch_y + eyepatch_height, frameh)
    eyepatch_x_end = min(eyepatch_x + eyepatch_width, framew)

    # Resize the eyepatch overlay to match the calculated dimensions
    glasses_resized = cv2.resize(pirate_eyepatch, (eyepatch_x_end - eyepatch_x, eyepatch_y_end - eyepatch_y))

    # Apply alpha blending for transparent overlay
    alpha_glasses = glasses_resized[:, :, 3] / 255.0

    # Overlay the eyepatch on the frame using alpha blending
    try:
        for c in range(0, 3):
            frame[eyepatch_y:eyepatch_y_end, eyepatch_x:eyepatch_x_end, c] = alpha_glasses * glasses_resized[:, :, c] + \
                                                                        (1 - alpha_glasses) * frame[
                                                                                            eyepatch_y:eyepatch_y_end,
                                                                                        eyepatch_x:eyepatch_x_end, c]
    except Exception as e:
        print("An exception occurred:", str(e))



    # BEARD
    # beard SIZE(adjust as need)
    beard_height = int(h * 1.5)
    beard_width = int(beard_height * pirate_beard.shape[1] / pirate_beard.shape[0])   # proportional to mustache_height

    # mustache_x and mustache_y are top left coordiantes of mustache overlay
    beard_x = upper_lip_x - int(beard_width / 2)      # Centers mustache horizonally on the upper lip
    beard_y = upper_lip_y - int(beard_height / 2)     # centers mustache vertically on upper lip

    # Ensure the mustache is within the frame
    beard_y = max(beard_y, 0)
    beard_x = max(beard_x, 0)
    beard_y_end = min(beard_y + beard_height, frameh)
    beard_x_end = min(beard_x + beard_width, framew)

    # resize mustache overlay to match calculated dimensions
    beard_resized = cv2.resize(pirate_beard, (beard_x_end-beard_x, beard_y_end-beard_y)) # note the updated size

    # Apply alpha blending for transparent overlay (needed for png transparency in imshow())
    alpha_beard = beard_resized[0:beard_y_end-beard_y, 0:beard_x_end-beard_x, 3] / 255.0

    # Overlay the hat on the frame using alpha blending
    # Overlay moving out of frame causes Exception error.
    try:
        for c in range(0, 3):
            frame[beard_y:beard_y + beard_height, beard_x:beard_x + beard_width,
            c] = alpha_beard * beard_resized[:, :, c] + (
                    1 - alpha_beard) * frame[beard_y:beard_y + beard_height,
                                          beard_x:beard_x + beard_width, c]
    except Exception as e:
        print("An exception occurred:", str(e))



    # Overlay the hat on top of the head
    hat_height = int(eye_distance * 3.3) # adjust this factor to get the desired hat size
    hat_width = int(hat_height * pirate_hat.shape[1] / pirate_hat.shape[0])

    # Ensure the hat is within the frame
    hat_x = top_head_x - int(hat_width / 2)
    hat_y = top_head_y - hat_height
    hat_y = max(hat_y, 0)
    hat_x = max(hat_x, 0)
    hat_y_end = min(hat_y + hat_height, frameh)
    hat_x_end = min(hat_x + hat_width, framew)

    # resize hat overlay to match calculated dimensions
    hat_resized = cv2.resize(pirate_hat, (hat_x_end-hat_x, hat_y_end-hat_y)) # note the updated size

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
    

    return frame
