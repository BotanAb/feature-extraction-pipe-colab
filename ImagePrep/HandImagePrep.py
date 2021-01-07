import glob
import importlib.util
import cv2

import pandas as pd
import numpy as np
import math
import copy
import os
from pathlib import Path
import shutil
import imutils
import matplotlib
import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from matplotlib.backends.backend_pdf import PdfPages

import skimage
from skimage import exposure
from skimage.util import invert
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.morphology import skeletonize
from tqdm import tqdm

#######################################################################################################
from Testing.logging.loggingTools import my_logger, my_timer


class HandImagePrep:

    def capture(self, path, idx, feature):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        j = idx
        i = 0
        count = 0
        to_gray = False  # Set to True if images have to saved in gray scale
        print(frame_count)
        # while count < int(frame_count):
        while count < 150:
            ret, frame = cap.read()
            print("Framerate: " + str(count))
            if ret:
                if not self.has_motion_blur(frame):
                    if to_gray:
                        frame = self.to_gray_scale(frame)

                    input_path = './ImagePrep/inputImages/' + feature + '/'

                    #if not os.path.exists(input_path):
                     #   os.makedirs(input_path)

                    cv2.imwrite(input_path + feature + str(j) + '-' + str(i) + '.jpg',
                                frame)
                    count += 10  # skip 9 frame
                else:
                    print('Frame ' + str(count + 1) + ' from ' + feature + ' is too blurry')
                    count += 1  # don't skip any frames if the current frame has motion blur
                cap.set(1, count)
                i += 1

            else:
                # cap.release()
                # cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()
        self.move_video(path, feature)

    def to_gray_scale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def check_if_cv2_exists(self):
        cv2_spec = importlib.util.find_spec('cv2')
        print(cv2_spec)
        # Attempting ffmpeg install if cv2 is not correctly installed

        if not cv2_spec:
            return False
        return True

    def check_if_ffmpeg_exists(self):
        try:
            os.system('ffmpeg')
            print("ffmpeg correctly installed")
        except:
            print(
                "Task terminated due to insufficient installed modules, please install cv2 or ffmpeg first and retry")
            ffmpeg_installed = False
            return ffmpeg_installed

    def has_motion_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = self.variance_of_laplacian(gray)

        # if the focus measure is less than the supplied threshold (10),
        # then the image should be considered "blurry"
        if fm < 5:
            return True
        return False

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def move_video(self, path, feature):
        Path("./Videos/usedVideos/" + feature).mkdir(parents=True, exist_ok=True)
        shutil.move(path, "./Videos/usedVideos/" + feature + '/' + feature + str(
            len(os.listdir("./Videos/usedVideos/" + feature)) + 1) + ".mp4")

    def vid_to_img(self, features, videodir):
        for feature in features:
            paths = glob.glob(videodir.get_dir_dict(feature),
                              recursive=True)  # kijkt in video + feature en zoekt alle mp4 bestanden (array van strings)
            print("capturing frames from " + feature)
            # print(paths)

            if paths:
                Path('./ImagePrep/inputImages/' + feature).mkdir(parents=True,
                                                   exist_ok=True)  # maakt nieuwe map aan met de featurenaam in IMGprep
                cv2_installed = self.check_if_cv2_exists()
                ffmpeg_installed = False
                if not cv2_installed:
                    ffmpeg_installed = self.check_if_ffmpeg_exists()
                for idx, path in enumerate(paths):
                    if cv2_installed:
                        self.capture(path, idx, feature)
                    elif ffmpeg_installed:
                        self.capture_ffmpeg(path, feature, idx)
                    else:
                        print("Er is niks geinstalleerd")
                        quit()
                        self.move_video(paths[idx], type)
            else:
                print("No videos found in map: " + feature)
        print("Done capturing frames")

    def capture_ffmpeg(self, path, feature, idx):
        cmd = 'ffmpeg -i ' + path + ' -f image2 "./ImagePrep/inputImages/' + feature + '/' + feature + str(
            idx) + '-' + '%05d' + '.jpg"'
        print(cmd)
        os.system(cmd)

    def calculate_fingers(self, res):
        #  convexity defect
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 5:
            defects = cv2.convexityDefects(res, hull)
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                if cnt > 0:
                    return cnt + 1
                else:
                    return 0
        return 0

    def do_smoothing(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        return cv2.bilateralFilter(img_rescale, 5, 50, 100)  # Smoothing

    def remove_background(self, image):
        background_removal = cv2.createBackgroundSubtractorMOG2(0, 50)
        fg_mask = background_removal.apply(image)
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        return cv2.bitwise_and(image, image, mask=fg_mask)

    def create_skin_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        skin_mask = cv2.inRange(hsv, lower, upper)
        return cv2.blur(skin_mask, (2, 2))

    def get_contours_and_hull(self, image):
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        skin_mask = copy.deepcopy(thresh)  # Thresholden
        contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contour
        return contours, hierarchy

    def move_images(self, paths, feature):
        i = 0
        Path("./ImagePrep/inputImages/usedImages/" + feature).mkdir(parents=True,
                                             exist_ok=True)  # maakt nieuwe map aan met de featurenaam in IMGprep
        for path in paths:
            print("moving: " + path + " to inputImages folder")
            shutil.move(path, "./ImagePrep/inputImages/usedImages/"+feature + "/" + feature + str(i) + ".jpg")
            i += 1

    #######################################################################################################
    def prep_images(self, features):
        print("======================== HAND IMAGE PREP ==========================================")
        for feature in features:
            paths = glob.glob('./ImagePrep/inputImages/' + feature + '/*.jpg', recursive=True)

            print("preparing " + feature + " images")
            j = 0
            for path in paths:

                outdir = "./ImageSim/inputImages/" + feature + "/"
                Path(outdir).mkdir(parents=True, exist_ok=True)
                oFnam = outdir + os.path.basename(path)

                #######################################################################################################
                #
                # Image preparation
                #
                #######################################################################################################
                print("Get " + path)
                # print(path)
                img = cv2.imread(path)

                # plt.imshow(img)
                # plt.show()

                smoothing = self.do_smoothing(img)
                # plt.imshow(smoothing)
                # plt.show()

                background_removal = self.remove_background(smoothing)
                # plt.imshow(background_removal)
                # plt.show()

                skin_mask = self.create_skin_mask(background_removal)
                # plt.imshow(skin_mask)
                # plt.show()

                contours, hierarchy = self.get_contours_and_hull(skin_mask)
                # print(contours)
                # print(hierarchy)

                length = len(contours)
                max_area = -1
                drawing = np.zeros(img.shape, np.uint8)

                if length > 0:

                    for index, c in enumerate(contours):
                        area = cv2.contourArea(c)

                        if area < 250:  # Om kleine area's te skippen
                            continue
                        if area > max_area:
                            max_area = area
                            max_c = c
                            res = contours[index]

                    if max_area > 0:
                        hull = cv2.convexHull(res)  # Make hull
                        count = 1  # calculateFingers(res, drawing)
                        if count > 0:  # Maak alleen skin als vingers er zijn
                            # print(count)

                            # Draw contours
                            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)  # Create bounding hull
                            cv2.drawContours(drawing, [res], 0, (255, 0, 0), 2)  # Create bounding hull

                            # Create bounding box
                            bounding_rect = cv2.boundingRect(max_c)
                            x = bounding_rect[0] - 100
                            y = bounding_rect[1] - 100
                            x_len = bounding_rect[2] + (2 * 100)
                            y_len = bounding_rect[3] + (2 * 100)
                            if y < 0:
                                y = 0
                            if x < 0:
                                x = 0

                            cv2.rectangle(img, (x, y),
                                          (x + x_len, y + y_len),
                                          (255, 255, 255),
                                          2)
                            if x_len > 400 or y_len > 400:
                                try:
                                    # Crop image
                                    crop_img = img[y:y + y_len,
                                               x:x + x_len]
                                    cv2.imwrite(oFnam, crop_img)
                                    j += 1
                                    # print("Save "+oFnam)
                                except:
                                    print("er gaat iets fout")
                # plt.imshow(img)
                # plt.show()

                # plt.imshow(drawing)
                # plt.show()

                # plt.imshow(crop_img)
                # plt.show()

                # cv2.imshow("drawing", drawing)
                # cv2.waitKey(0)

            self.move_images(paths, feature)


#######################################################################################################


if __name__ == '__main__':
    # features = ["fist", "palm", "thumb"]
    image_prep = HandImagePrep()
    # imageprep.vidToImg(features)
    # imageprep.prepHandImages(features)
    # imageprep.checkIfCv2Exists()