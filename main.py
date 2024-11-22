import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

cam_quit = False

while cam_quit == False:
    rval, frame = vc.read()
    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        cam_quit = True

cv2.destroyWindow("preview")
vc.release()