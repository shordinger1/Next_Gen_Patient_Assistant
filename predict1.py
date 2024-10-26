import os
import time
import numpy as np
from PIL import Image
from siamese1 import Siamese

if __name__ == "__main__":
    model = Siamese()

    # Path to the compare folder
    compare_dir = 'datasets/Split_smol/compare'

    # Load all comparison images from the compare folder into a dictionary
    compare_images = {}
    for filename in os.listdir(compare_dir):
        class_name = os.path.splitext(filename)[0]  # Get the class name from the filename
        compare_images[class_name] = Image.open(os.path.join(compare_dir, filename))

    # Validation set directory
    input_dir = 'datasets/Split_smol/val'

    rightnumall = 0
    falsenumall = 0
    starttime = time.time()

    for class_folder in os.listdir(input_dir):
        print(f"Evaluating class: {class_folder}")

        rightnum = 0
        falsenum = 0

        # Get the corresponding comparison image for this class
        comparison_image = compare_images.get(class_folder)
        if comparison_image is None:
            print(f"No comparison image found for class: {class_folder}")
            continue

        # Iterate over each image in the current validation class folder
        class_path = os.path.join(input_dir, class_folder)
        for filename in os.listdir(class_path):
            image_jc = Image.open(os.path.join(class_path, filename))

            # Detect similarity with the comparison image
            probability = model.detect_image(comparison_image, image_jc)

            # Assuming the model returns a similarity score and we're thresholding at 0.5
            if probability > 0.5:
                rightnum += 1
            else:
                falsenum += 1

        print(f"Right: {rightnum}, False: {falsenum}")
        rightnumall += rightnum
        falsenumall += falsenum
        print('Current class accuracy:')
        print(rightnum / (rightnum + falsenum) if (rightnum + falsenum) > 0 else 0)

    print('Overall validation accuracy:')
    print(rightnumall / (rightnumall + falsenumall) if (rightnumall + falsenumall) > 0 else 0)
    print('Total number of images evaluated:')
    print(rightnumall + falsenumall)

    endtime = time.time()
    print('Total time taken:')
    print(endtime - starttime)
    print('Average time per image:')
    print((endtime - starttime) / (rightnumall + falsenumall) if (rightnumall + falsenumall) > 0 else 0)
