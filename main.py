import cv2
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np

class Train_Card:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"

vc = cv2.VideoCapture(0)

# load cards
train_cards = []
directory = os.fsencode('Cards/')

i = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        train_cards.append(Train_Card())
        train_cards[i].name = filename[0:-4]

        card_image = cv2.imread(f'Cards/{filename}', cv2.IMREAD_GRAYSCALE)
        card_image = cv2.resize(card_image, (250, 350)) 
        train_cards[i].img = card_image
        i = i + 1

# Keep scanning webcam while application is on
cam_quit = False
while cam_quit == False:

    # Read from camera
    # Display video
    rval, frame = vc.read()

    # find card
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur, 1, 50)

    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0];

    cards_contours = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 250: 
            cards_contours.append(c)

    cv2.drawContours(frame, cards_contours, -1, (0, 255, 0), 3) 

    if(len(contours)>0):
        max_cards_contour = max(contours, key = cv2.contourArea)
        
        max_cards_perimeter = cv2.arcLength(max_cards_contour, True);
        approx = cv2.approxPolyDP(max_cards_contour,0.01*max_cards_perimeter,True)
        pts = np.float32(approx)
        
        if(len(pts) >= 4):
            max_height = 350
            max_width = 250

            # four input point 
            input_pts=np.float32([pts[0][0],
                                pts[1][0],
                                pts[3][0],
                                pts[2][0],])
            
            # output points for new transformed image
            output_pts = np.float32([[0, 0],
                                    [0, max_height],
                                    [max_width , 0],
                                    [max_width , max_height]])


            # Compute the perspective transform M
            M = cv2.getPerspectiveTransform(input_pts,output_pts)

            focused_card = cv2.warpPerspective(gray,M,(max_width,max_height),flags=cv2.INTER_LINEAR)
            cv2.imshow("image",focused_card)

            best_match_card_diff = 10000000000000
            best_match_card_name = "Unknown"

            for card in train_cards:
                diff_img = cv2.absdiff(focused_card, card.img)
                diff = int(np.sum(diff_img)/255)
                if diff < best_match_card_diff:
                        best_match_card_diff = diff
                        best_match_card_name = card.name

            print(best_match_card_name)

    # show images
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)
    cv2.imshow("edges", edges)
    cv2.imshow("preview", frame)

    # exit when ESC is pressed
    key = cv2.waitKey(20)
    if key == 27:
        cam_quit = True

cv2.destroyWindow("preview")
vc.release()