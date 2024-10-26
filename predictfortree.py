import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from siamese1 import Siamese

if __name__ == "__main__":
    model = Siamese()

    # Path to the compare folder
    compare_dir = 'datasets/Split_smol/compare'

    # Load all comparison images from the compare folder into a dictionary
    compare_images = {}
    compare_classes = []
    for filename in os.listdir(compare_dir):
        class_name = os.path.splitext(filename)[0]  # Get the class name from the filename
        compare_images[class_name] = Image.open(os.path.join(compare_dir, filename))
        compare_classes.append(class_name)

    # Validation set directory
    input_dir = 'datasets/Split_smol/train'

    results = []  # To store the results

    starttime = time.time()

    for class_folder in os.listdir(input_dir):
        print(f"Evaluating class: {class_folder}")

        # Path to the current validation class folder
        class_path = os.path.join(input_dir, class_folder)

        # Iterate over each image in the current validation class folder
        for filename in os.listdir(class_path):
            image_jc = Image.open(os.path.join(class_path, filename))

            # Store the results for the current image
            row = {
                'image_name': filename,
                'true_class': class_folder
            }

            # Compare with each of the 8 compare images
            for compare_class, compare_image in compare_images.items():
                probability = model.detect_image(compare_image, image_jc)
                row[f'similarity_with_{compare_class}'] = probability  # Save similarity score

            # Append the row to the results list
            results.append(row)

    # Convert the results to a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Save the results to a CSV file
    output_file = 'similarity_results.csv'
    df_results.to_csv(output_file, index=False)

    print(f'Results saved to {output_file}')

    endtime = time.time()
    print('Total time taken:')
    print(endtime - starttime)