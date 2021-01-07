import glob
import sys
import traceback
from ImagePrep.HandImagePrep import HandImagePrep
from ImageSim.ImageSim import ImageSim
from ConvNeuralNetwork.CNN import CNN
from ConvNeuralNetwork.Classes.Directories import Directory
# from errorHandling import generic_exception_handler

def main():
    features = ["fist", "palm", "thumb"]
    video_dir = Directory(
        True)  ## True uses hardcoded directories from earlier versions, False lets user specify own directory

    if video_dir.get_standard():
        video_dir.set_standard_directories(features)
    else:
        video_dir.set_new_directories(features)

    prep_image = HandImagePrep()
    prep_image.vid_to_img(features, video_dir)
    prep_image.prep_images(features)
    image_sim = ImageSim()
    image_sim.sim_images(features)
    cnn = CNN(features=features)
    cnn.train_cnn()

    print("Einde programma")

    predict_dir = Directory(
        True)  ## True uses hardcoded directories from earlier versions, False lets user specify own directory
    if predict_dir.get_standard():
        predict_dir.set_standard_directories(features)
    else:
        predict_dir.set_new_directories(features)

    print(cnn.predict(imagePath="./Testing/testImages/fist6.jpg", features=features))
    cnn.predict_multiple("./Testing/testImages/test2.jpg", features)
    print(cnn.predict('./Testing/testImages/fist6.jpg', features))
    print(cnn.predict('./Testing/testImages/palm18.jpg', features))
    print(cnn.predict('./Testing/testImages/thumb19.jpg', features))

    cnn.predict_multiple("./Testing/testImages/old.jpg", features)
    print("einde programma")


if __name__ == '__main__':
    main()
