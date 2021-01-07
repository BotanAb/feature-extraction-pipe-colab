import unittest
import glob
import cv2
import logging

from PIL import Image as PImage

from ImagePrep.HandImagePrep import HandImagePrep
from Testing.logging.loggingTools import my_timer, my_logger


class TestHandPrep(unittest.TestCase):

    def test_check_if_cv2_exists(self):
        prep_image = HandImagePrep()
        self.assertTrue(prep_image.check_if_cv2_exists())

    def test_check_if_ffmpeg_exists(self):
        prep_image = HandImagePrep()
        self.assertTrue(prep_image.check_if_ffmpeg_exists())

    def test_false_check_if_ffmpeg_exists(self):
        prep_image = HandImagePrep()
        self.assertRaises(Exception, prep_image.check_if_ffmpeg_exists())

    def test_has_motion_blur(self):
        prep_image = HandImagePrep()
        cap = cv2.VideoCapture('../testImages/fist0113.jpg')
        ret, frame = cap.read()
        if ret:
            if not prep_image.has_motion_blur(frame):
                print('no blur')
            else:
                print('blurry')
