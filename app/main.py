import cv2
from processing import process_image


def image_capture():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error opening camera")
        return False, None

    ret = cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    ret &= cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not ret:
        print("Error setting resolution")
        return False, None

    ret, frame = cam.read()

    cam.release()

    return ret, frame


def main():

    #ret, frame = image_capture()
    #if not ret:
    #    return
    #cv2.imwrite("001-captured-frame.jpg", frame)

    frame = cv2.imread("001-captured-frame.jpg")
    output_array = process_image(frame, debug=True)
    print(f"Processed array shape: {output_array.shape}")


if __name__ == "__main__":
    main()
