import os
import cv2
import yaml
import json
import random
import argparse
import datetime

from tqdm import tqdm

class_count = dict()

def xcycwhToxyxy(xcycwh, img_width: int, img_height: int) -> None:
    """
    Convert YOLO coordinates to COCO coordinates.

    Args:
    - yolo_coords: Tuple (x_center, y_center, width, height) in YOLO format (normalized)
    - img_width: Width of the image
    - img_height: Height of the image

    Returns:
    - Tuple (x_min, y_min, width, height) in COCO format
    """
    x_center, y_center, width, height = xcycwh

    # Convert normalized values to absolute values
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Calculate x_min and y_min
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)

    (x_min, y_min, width_abs, height_abs) = int(x_min), int(y_min), int(width_abs), int(height_abs)
    return (x_min, y_min, width_abs, height_abs)

def process_dataset(class_names: str, dataset: str, new_dataset_dir: str) -> None:
    """
    This function processes a dataset of images and their corresponding labels,
    crops the images based on the provided labels, and saves the cropped images
    in a new directory structure.

    Parameters:
    - class_names (list): A list of class names corresponding to the labels in the dataset.
    - dataset (str): The path to the dataset directory.
    - new_dataset_dir (str): The path to the new directory where the processed dataset will be saved.

    Returns:
    None
    """
    dataset_types = ['train', 'valid', 'test']
    images = list()

    new_dataset_dir = os.path.join(new_dataset_dir, 'images')

    dataset_name = os.path.basename(dataset_dir).split('_dataset')[0]

    if not dataset_name in class_count:
        class_count[dataset_name] = dict()

    for class_name in class_names:
        if not class_name in class_count[dataset_name]:
            class_count[dataset_name][class_name] = 0
        os.makedirs(os.path.join(new_dataset_dir, dataset_name, class_name), exist_ok=True)

    for dataset_type in dataset_types:
        images_path = os.path.join(dataset, dataset_type, 'images')
        images.extend([os.path.join(images_path, image) for image in os.listdir(images_path)])

    for image_path in tqdm(images, total=len(images), desc=f'Creating dataset: {os.path.basename(dataset)}', disable=True):
        label_path = os.path.join(image_path.split('images')[0], 'labels', f'{os.path.basename(image_path).split(".jpg")[0]}.txt')

        if os.path.exists(image_path) and os.path.exists(label_path):
            image = cv2.imread(image_path)
            h, w = image.shape[:2]

            # Load labels
            with open(label_path, 'r') as file:
                labels_yolo = file.readlines()

            try:
                for label_yolo in labels_yolo:
                    label = label_yolo.strip().split()
                    x_center, y_center, width, height = map(float, label[1:])

                    # Convert YOLO coordinates to COCO format
                    x_min, y_min, width, height = xcycwhToxyxy((x_center, y_center, width, height), w, h)

                    # cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 0, 255), 2)
                    cropped_image = image[y_min : y_min + height, x_min : x_min + width]

                    class_count[dataset_name][class_names[int(label[0])]] += 1
                    cropped_image_path = os.path.join(new_dataset_dir, dataset_name, class_names[int(label[0])], f"{class_count[dataset_name][class_names[int(label[0])]]}.jpg")

                    if cropped_image.shape[0] <= 25 or cropped_image.shape[1] <= 25:
                        continue

                    if class_names[int(label[0])] == 'coca_cola_diet' and class_count[dataset_name][class_names[int(label[0])]] == 5:
                        print(cropped_image.shape)

                    cv2.imwrite(cropped_image_path, cropped_image)
            except Exception as e:
                print(f'Error processing image: {image_path}, error: {e}')
        else:
            print(f'Label file not found for: {label_path.split(dataset)[-1]}')

def create_labels(new_dataset_dir: str) -> None:
    """
    This function generates label files for each product image in the dataset.
    The label files contain the paths to the corresponding images.

    Parameters:
    - new_dataset_dir (str): The path to the new dataset directory where the images and labels will be saved.

    Returns:
    None
    """
    images_path = os.path.join(new_dataset_dir, 'images')
    labels_path = os.path.join(new_dataset_dir, 'labels')

    categories = os.listdir(images_path)
    for category in tqdm(categories, desc='Generating labels'):
        category_path = os.path.join(images_path, category)
        for product in os.listdir(category_path):
            labels = os.listdir(os.path.join(category_path, product))
            os.makedirs(os.path.join(labels_path, category), exist_ok=True)
            with open(os.path.join(labels_path, category, f'{product}.txt'), 'w') as file:
                for label in labels:
                    file.write(f"{os.path.join(images_path.split('dataset/')[-1], category, product, label)}\n")

def split_dataset(new_dataset_dir: str) -> None:
    """
    This function splits the dataset into training, validation, and testing sets.
    It reads the labels from the 'labels' directory, shuffles them, and then splits them into three separate lists.
    The function then writes these lists to separate text files in the 'labels' directory.

    Parameters:
    - new_dataset_dir (str): The path to the new dataset directory where the images and labels are saved.

    Returns:
    - None
    """
    labels_path = os.path.join(new_dataset_dir, 'labels')

    train_labels = list()
    valid_labels = list()
    test_labels = list()

    categories = os.listdir(labels_path)
    categories = [file for file in os.listdir(labels_path) if os.path.isdir(os.path.join(labels_path, file))]
    for category in tqdm(categories, desc='Spliting dataset'):
        label_files = os.listdir(os.path.join(labels_path, category))
        for label_file in label_files:
            with open(os.path.join(labels_path, category, label_file)) as file:
                labels = file.readlines()
                random.shuffle(labels)
                train_count = int(len(labels) * 0.8)
                valid_count = int(len(labels) * 0.1)

                train_labels.extend([label.strip() for label in labels[:train_count]])
                valid_labels.extend([label.strip() for label in labels[train_count:train_count + valid_count]])
                test_labels.extend([label.strip() for label in labels[train_count + valid_count:]])

    with open(os.path.join(labels_path, 'train.txt'), 'w') as train_file:
        random.shuffle(train_labels)
        for line in train_labels:
            train_file.write(f'{line}\n')
    with open(os.path.join(labels_path, 'valid.txt'), 'w') as valid_file:
        random.shuffle(valid_labels)
        for line in valid_labels:
            valid_file.write(f'{line}\n')
    with open(os.path.join(labels_path, 'test.txt'), 'w') as test_file:
        random.shuffle(test_labels)
        for line in test_labels:
            test_file.write(f'{line}\n')

def create_model_files(new_dataset_dir, model_data_dir: str) -> None:
    # get current date in YYYYMMDD format
    date = datetime.datetime.now().strftime("%Y%m%d")

    model_data_dir = os.path.join(model_data_dir, date)
    os.makedirs(model_data_dir, exist_ok=True)

    # Create Hierarchical JSON file
    hierarchical_data = {
        "L0": [
            {"name": "product"}
        ],
        "L1": [],
        "L2": [],
    }

    for index, category in enumerate(os.listdir(os.path.join(new_dataset_dir, 'images'))):
        l1Dict = {}
        l1Dict["name"] = category
        l1Dict["id"] = str(index)
        l1Dict["parent"] = "product"
        hierarchical_data["L1"].append(l1Dict)

        for productId, product in enumerate(os.listdir(os.path.join(new_dataset_dir, 'images', category))):
            l2Dict = {}
            l2Dict["name"] = product
            l2Dict["id"] = str(productId)
            l2Dict["parent"] = category
            hierarchical_data["L2"].append(l2Dict)

    with open(os.path.join(model_data_dir, "hierarchical_representation.json"),"w") as jsonFile:
        json.dump(hierarchical_data, jsonFile, indent = 4)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_dir", default='/home/nammathalle/work/retail_product_recognition_system/model/original_dataset', required=False, help="Path to the configuration file")
    parser.add_argument("--new_dataset_dir", default='/home/nammathalle/work/retail_product_recognition_system/model/dataset', required=False, help="Path to the configuration file")
    parser.add_argument("--model_data_dir", default='/home/nammathalle/work/retail_product_recognition_system/model/model_data', required=False, help="Path to the model directory")
    args = parser.parse_args()

    original_dataset_dir = args.original_dataset_dir
    new_dataset_dir = args.new_dataset_dir
    model_data_dir = args.model_data_dir
    
    # Check if the original dataset directory exists
    if not os.path.isdir(original_dataset_dir):
        print(f"Error: {original_dataset_dir} is not a valid directory.")
        exit(1)

    # Create images and labels directory 
    os.makedirs(os.path.join(new_dataset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_dir, 'labels'), exist_ok=True)

    datasets = [folder for folder in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, folder))]
    
    for dataset in tqdm(datasets, desc='Generating dataset'):
        
        config_file = os.path.join(original_dataset_dir, dataset, 'data.yaml')
        # Check if the config file exists
        if os.path.isfile(config_file):
            class_names = list()
            dataset_dir = list()
            try:
                # print(f'Found YoloV7 dataset in: {dataset}')
                # open config file
                with open(config_file, "r") as file:
                    config = yaml.safe_load(file)
                    
                class_names = config.get('names')

                dataset_dir = os.path.join(original_dataset_dir, dataset)
            except Exception as e:
                print(f'Error parsing config file: {config_file}. Error: {str(e)}')
                print(f'Skipping: {dataset}')
                continue

            process_dataset(class_names, dataset_dir, new_dataset_dir)
            
        else:
            print(f'No YoloV7 dataset found, skipping: {dataset}')
    
    create_labels(new_dataset_dir)

    split_dataset(new_dataset_dir)

    create_model_files(new_dataset_dir, model_data_dir)

    # TODO: Using matplotlib plot class count of each class in each category
