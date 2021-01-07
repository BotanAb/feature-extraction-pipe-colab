import unittest
from unittest.mock import patch, call

from ImageSim.ImageSim import ImageSim
from ImagePrep.HandImagePrep import HandImagePrep

class TestImageSim(unittest.TestCase):

    def test_move_image_failed(self):
        sim_image = ImageSim()
        message = "arg"
        self.assertFalse(sim_image.sim_images(["test1", "test2", "test3"]), message)
