import os
from PIL import Image

# Define paths
image_dir = 'data/images'
label_dir = 'runs/detect/exp3/labels'
output_dir = './cropped_images/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def cut_image(img_result):
    img=image = Image.fromarray(img_result[0])
    label=img_result[1]
    class_id, x_center, y_center, width, height = map(float, label)
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    box_width_abs = width * img_width
    box_height_abs = height * img_height
    x_min = int(x_center_abs - (box_width_abs / 2))
    y_min = int(y_center_abs - (box_height_abs / 2))
    x_max = int(x_center_abs + (box_width_abs / 2))
    y_max = int(y_center_abs + (box_height_abs / 2))
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    return cropped_img




# Get list of image files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    # Open image
    image_path = os.path.join(image_dir, image_file)
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Corresponding label file
    label_file = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))

    # Check if the label file exists
    if not os.path.exists(label_file):
        print(f"Label file does not exist for image {image_file}, skipping...")
        continue  # Skip to the next image if the label file doesn't exist

    # Read label file
    with open(label_file, 'r') as f:
        labels = f.readlines()

    # Process each detected object
    for idx, label in enumerate(labels):
        class_id, x_center, y_center, width, height = map(float, label.split())

        # Convert normalized coordinates to absolute pixel values
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        box_width_abs = width * img_width
        box_height_abs = height * img_height

        # Calculate bounding box (top-left and bottom-right corners)
        x_min = int(x_center_abs - (box_width_abs / 2))
        y_min = int(y_center_abs - (box_height_abs / 2))
        x_max = int(x_center_abs + (box_width_abs / 2))
        y_max = int(y_center_abs + (box_height_abs / 2))

        # Crop the object from the image
        cropped_img = img.crop((x_min, y_min, x_max, y_max))

        # Save the cropped image
        cropped_img_name = f"{image_file.replace('.jpg', '')}_obj{idx}.jpg"
        cropped_img.save(os.path.join(output_dir, cropped_img_name))

        print(f"Cropped object {idx} from {image_file} and saved as {cropped_img_name}")
