import cv2

import numpy as np
import copy

import matplotlib.pyplot as plt

from skimage import exposure
from PIL import Image


def do_smoothing(image):
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return cv2.bilateralFilter(img_rescale, 5, 50, 100)  # Smoothing


def remove_background(image):
    backgroundremoval = cv2.createBackgroundSubtractorMOG2(0, 50)
    fgmask = backgroundremoval.apply(image)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    return cv2.bitwise_and(image, image, mask=fgmask)


def create_skinmask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 48, 80], dtype="uint8")
    lower = np.array([0, 30, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    # plt.imshow(skinMask)
    # plt.show()
    return cv2.blur(skinMask, (2, 2))


def get_contours_and_hull(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    skinMask1 = copy.deepcopy(thresh)  # Thresholden
    contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contour
    return contours, hierarchy


#######################################################################################################
def prep_bounding_boxes(path):
    #######################################################################################################
    #
    # Image preparation
    #
    #######################################################################################################
    # print(path)
    img = cv2.imread(path)

    # plt.imshow(img)
    # plt.show()

    smoothing = do_smoothing(img)
    # plt.imshow(smoothing)
    # plt.show()

    background_removal = remove_background(smoothing)
    # plt.imshow(background_removal)
    # plt.show()

    skin_mask = create_skinmask(background_removal)
    # plt.imshow(skinMask)
    # plt.show()

    contours, hierarchy = get_contours_and_hull(skin_mask)
    # print(contours)
    # print(hierarchy)

    len_contours = len(contours)
    drawing = np.zeros(img.shape, np.uint8)

    img_height, img_width, _ = img.shape

    PX_RATIO = 0.04

    if len_contours > 0:
        area_array = []

        for index, c in enumerate(contours):
            area = cv2.contourArea(c)
            area_array.append(area)

        sorted_contours = sorted(zip(area_array, contours), key=lambda x: x[0], reverse=True)

        top_contours = sum(1 for x in contours if
                           (cv2.boundingRect(x)[2] * cv2.boundingRect(x)[3]) / (img_width * img_height) > PX_RATIO)
        print(top_contours)

        bounding_rects = []

        for i in range(top_contours):
            contour = sorted_contours[i][1]

            hull = cv2.convexHull(contour)  # Make hull

            # Draw contours
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)  # Create bounding hull
            cv2.drawContours(drawing, [contour], 0, (255, 0, 0), 2)  # Create bounding hull

            # Create bounding box
            bounding_rect = cv2.boundingRect(contour)
            bounding_rects.append(bounding_rect)

        return bounding_rects


def getImageFromBoundingBox(path, bounding_rect):
    PX_RATIO = 0.1

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_height, img_width, _ = img.shape

    width_margin = img_width * PX_RATIO
    height_margin = img_height * PX_RATIO

    bounding_x = bounding_rect[0] - int((width_margin / 2))
    bounding_y = bounding_rect[1] - int((height_margin / 2))
    bounding_width = bounding_rect[2] + int(width_margin)
    bounding_height = bounding_rect[3] + int(height_margin)

    if bounding_x < 0:
        bounding_x = 0
    if bounding_y < 0:
        bounding_y = 0

    crop_img = img[bounding_y:bounding_y + bounding_height, bounding_x:bounding_x + bounding_width]
    # plt.axis('off')
    # plt.imshow(crop_img)
    # plt.show()

    # plt.imshow(crop_img)
    # plt.savefig(fname="./image" + str(bounding_x) + str(bounding_y) + str(bounding_width) + str(bounding_height) + ".png", dpi=200)

    return Image.fromarray(crop_img)


def renderImageWithBoundingBox(path, bounding_rects, predicted_items):
    MARGIN = 50  # for when text is outside of the original image's dimensions

    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT)

    unique_items = list(set(predicted_items))
    box_colours = {}

    for i in unique_items:
        colour = list(np.random.choice(range(255), size=3))
        box_colours[i] = colour

    print(box_colours)

    for index, bounding_rect in enumerate(bounding_rects):
        bounding_x = bounding_rect[0] + MARGIN
        bounding_y = bounding_rect[1] + MARGIN
        bounding_width = bounding_rect[2]
        bounding_height = bounding_rect[3]

        if bounding_x < 0:
            bounding_x = 0
        if bounding_y < 0:
            bounding_y = 0

        current_colour = tuple(int(c) for c in box_colours.get(predicted_items[index]))
        print(current_colour)

        cv2.rectangle(img, (bounding_x, bounding_y),
                      (bounding_x + bounding_width, bounding_y + bounding_height),
                      current_colour,
                      4)
        cv2.putText(img, predicted_items[index], (bounding_x, bounding_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    current_colour, 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.show()
