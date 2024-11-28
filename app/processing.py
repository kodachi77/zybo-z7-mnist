import os
import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def is_within(box1, box2):
    """
    Check if box2 is fully within or overlaps with box1.
    box format: (x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x_min, box1_y_min = x1, y1
    box1_x_max, box1_y_max = x1 + w1, y1 + h1

    box2_x_min, box2_y_min = x2, y2
    box2_x_max, box2_y_max = x2 + w2, y2 + h2

    is_within = (
        box1_x_min <= box2_x_min <= box1_x_max
        and box1_y_min <= box2_y_min <= box1_y_max
        and box1_x_min <= box2_x_max <= box1_x_max
        and box1_y_min <= box2_y_max <= box1_y_max
    )

    overlaps = not (
        box1_x_max < box2_x_min
        or box1_x_min > box2_x_max
        or box1_y_max < box2_y_min
        or box1_y_min > box2_y_max
    )

    return is_within or overlaps


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def recognize_page(image, threshold1=50, threshold2=150, poly_epsilon=0.02):
    # Load the image and compute the ratio of the old height to the new height,
    # clone it, and resize it
    ratio = image.shape[0] / 480.0
    original_image = image.copy()
    image = image_resize(image, height=480)

    # convert the B&W image and apply edge detection
    k, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    edged = cv2.Canny(binary, threshold1, threshold2)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 2, "Contours should be a tuple with 2 elements"
    cnts = cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, poly_epsilon * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt:
        # apply the four point transform to obtain a top-down view of the
        # original image
        return k, four_point_transform(original_image, screenCnt.reshape(4, 2) * ratio)
    else:
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edged, 1, np.pi / 180, 200)

        # Calculate the dominant vertical angle
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if 170 < angle <= 180:
                    angle = 180 - angle

                # Only consider near-horizontal or near-vertical lines
                if 80 < angle < 100:
                    angles.append(angle - 90)  # Adjust to vertical

        # Average angle for deskewing
        if len(angles) > 0:
            average_angle = np.mean(angles)
        else:
            average_angle = 0

        # Rotate the image
        (h, w) = original_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, average_angle, 1.0)
        return k, cv2.warpAffine(
            original_image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


def process_image(image, line_cnt=15, debug=False):
    """
    Detect numbers in the image, handle deskewing, and return them as a NumPy array.

    Our images are 720x1280, and the numbers will be 75-100 pixels in height.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (h, w) = image.shape
    max_dim = max(h, w)
    number_height = max_dim // line_cnt
    morph_kernel_size = number_height // 5

    # Deskew the image
    k, deskewed_image = recognize_page(image)
    if deskewed_image is None:
        return None

    # Threshold the deskewed image
    _, binary = cv2.threshold(deskewed_image, k, 255, cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )

    # Applying dilation on the threshold image
    dilation = cv2.dilate(binary, rect_kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # List to hold bounding boxes and their corresponding contours
    bounding_boxes = []

    min_size = 0.2 * number_height
    max_size = 2.5 * number_height

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small contours that are likely noise
        if min_size < w < max_size and min_size < h < max_size:
            skip = False
            for b in bounding_boxes:
                if is_within(b, (x, y, w, h)):
                    skip = True
                    break

            if not skip:
                if debug:
                    cv2.rectangle(deskewed_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
                bounding_boxes.append((x, y, w, h))
        else:
            if debug:
                print(f"Skipping {x, y, w, h}")
                cv2.rectangle(deskewed_image, (x, y), (x + w, y + h), (128, 128, 128), 2)

    # Sort bounding boxes top-to-bottom, then left-to-right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

    output_dir = "debug_images"
    if debug:
        os.makedirs(output_dir, exist_ok=True)
        # delete all old debug images
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    # List to store resized images
    processed_numbers = []

    i = 0
    for x, y, w, h in bounding_boxes:
        # Crop the number
        number_image = binary[y : y + h, x : x + w]
        #number_image = cv2.bitwise_not(number_image)
        resized = cv2.resize(number_image, (28, 28), interpolation=cv2.INTER_AREA)

        if debug:
            output_path = os.path.join(output_dir, f"dbg-{i + 1}.jpg")
            cv2.imwrite(output_path, resized)

        # Flatten to a 1D array
        flattened = resized.flatten()

        # Append to the list
        processed_numbers.append(flattened)
        i = i + 1

    if debug:
        cv2.imwrite(os.path.join(output_dir, "dbg-final.jpg"), deskewed_image)

    # Convert to NumPy array with dtype uint8
    result = np.array(processed_numbers, dtype=np.uint8)

    return result
