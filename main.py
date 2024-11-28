import cv2
import os
import numpy as np
import requests
import json
import re

LARGE = 10000000000
MAX_HEIGHT = 350
MAX_WIDTH = 250

class CardData:
    def __init__(self):
        self.image = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"
        self.code = "OP##-###"
        self.price = 0.00

def get_card_prices():

    # loop through card price url and grab all the products
    current_cursor = 0
    product_list = []
    while current_cursor != None:
        card_price_url = f"https://www.pricecharting.com/console/one-piece-two-legends?cursor={current_cursor}&format=json"
        response = requests.get(card_price_url).content
        response_dict = json.loads(response) 
        product_list = product_list + response_dict["products"]

        if "cursor" in response_dict:
            current_cursor = response_dict["cursor"]
        else:
            current_cursor = None
    card_info_dict = {}

    # loop through products and add the name and price into a dictionary
    for product in product_list:
        product_name = product["productName"]
        code_matches = re.findall(r"OP\d\d-\d\d\d", product_name)

        if len(code_matches) == 0:
            continue

        card_code = code_matches[0]
        card_name = product_name.replace(f" {card_code}", '')
        card_price = product["price1"]

        card_info_dict[card_code] = {
            "name": card_name,
            "price": card_price
        }
    
    return card_info_dict


def load_cards():
    new_card_list = []

    # get the card prices/name dictionary
    card_price_info_dict = get_card_prices()

    # read from card directory
    directory = os.fsencode('Cards/')

    # loop through all card photos
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            card_code = filename[0:-4]

            new_card = CardData()

            # get card code
            new_card.code = card_code
            
            if card_code not in card_price_info_dict:
                continue

            # get card image
            card_image = cv2.imread(f'Cards/{filename}', cv2.IMREAD_GRAYSCALE)
            card_image = cv2.GaussianBlur(card_image,(5,5),0)
            card_image = cv2.resize(card_image, (MAX_WIDTH, MAX_HEIGHT)) 
            new_card.image = card_image
            
            # get card name
            new_card.name = card_price_info_dict[card_code]["name"]

            # get card price
            new_card.price = card_price_info_dict[card_code]["price"]

            # add to list
            new_card_list.append(new_card)

    return new_card_list

def find_all_card_contours(edges_image):
    contours = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0];
    cards_contours = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 250: 
            cards_contours.append(c)

    return cards_contours

def find_input_card_points(edges_image):
    # find all cards in the image
    cards_contours = find_all_card_contours(edges_image)

    # return early if no cards were found
    if(len(cards_contours) == 0):
        return np.float32()

    # assume the largest found card is the card we are trying to scan
    max_card_contour = max(cards_contours, key = cv2.contourArea)

    # find the corners of the card
    max_card_perimeter = cv2.arcLength(max_card_contour, True)
    card_corners = cv2.approxPolyDP(max_card_contour,0.01*max_card_perimeter,True)
    card_corner_points = np.float32(card_corners)

    # return early if the found card is not card shape
    if(len(card_corner_points) != 4):
        return np.float32()

    # where the four corners are in the input
    input_points=np.float32([card_corner_points[0][0],
                        card_corner_points[1][0],
                        card_corner_points[3][0],
                        card_corner_points[2][0],])
    
    return input_points

def get_scanned_card(input_points, gray_image):
    # where we want the four corners to be in the output
    output_points = np.float32([[0, 0],
                            [0, MAX_HEIGHT],
                            [MAX_WIDTH , 0],
                            [MAX_WIDTH , MAX_HEIGHT]])

    # Compute the perspective transform 
    perspective_transform = cv2.getPerspectiveTransform(input_points,output_points)

    # transform the card to the correct dimensions
    tranformed_card = cv2.warpPerspective(gray_image,perspective_transform,(MAX_WIDTH,MAX_HEIGHT),flags=cv2.INTER_LINEAR)

    # Search through all cards and find the best match
    best_match_card_diff = LARGE
    best_match_card = None
    for card in card_list:
        diff_image = cv2.absdiff(tranformed_card, card.image)
        match_card_diff = int(np.sum(diff_image)/255)
        if match_card_diff < best_match_card_diff:
                best_match_card_diff = match_card_diff
                best_match_card = card

    cv2.imshow("image",tranformed_card)

    return best_match_card

vc = cv2.VideoCapture(0)
card_list = load_cards()
cam_quit = False

# Keep scanning webcam while application is on
while cam_quit == False:

    # exit when ESC is pressed
    key = cv2.waitKey(20)
    if key == 27:
        cam_quit = True

    # Read from camera
    _, frame = vc.read()

    # Show frame
    cv2.imshow("preview", frame)

    # process video to make scanning cards easier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur, 1, 50)

    # find card object in video
    input_card_points = find_input_card_points(edges)

    if(input_card_points.any() == False):
        continue
        
    # determine what card is in the video
    scanned_card = get_scanned_card(input_card_points, gray)

    # put text on the video
    if(scanned_card):
        print(f"{scanned_card.name}, {scanned_card.price}")

cv2.destroyWindow("preview")
vc.release()