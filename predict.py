import os
import time
import numpy as np
from PIL import Image
from siamese1 import Siamese
model = Siamese()
# Path to the compare folder (containing five images)
compare_dir = 'datasets/Split_smol/compare'
# Load all comparison images from the compare folder into a dictionary
compare_images = {}
for filename in os.listdir(compare_dir):
    class_name = os.path.splitext(filename)[0]  # Get the class name from the filename
    compare_images[class_name] = Image.open(os.path.join(compare_dir, filename))
# Track the best match


def predict(input_image):
    best_class = None
    highest_similarity = -1
    for class_name, comparison_image in compare_images.items():
        probability = model.detect_image(comparison_image, input_image)

        print(f"Similarity with {class_name}: {probability}")

        # Update the best match if this comparison is the most similar
        if probability > highest_similarity:
            highest_similarity = probability
            best_class = class_name

    # Output the class with the highest similarity score
    if best_class is not None:
        print(f"The input image most likely belongs to class: {best_class} with a similarity of {highest_similarity}")
    else:
        print("No matching class found.")
    return [best_class,highest_similarity]
    

