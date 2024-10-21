import queue
import sys
import threading
from glob import glob
import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import find_classes
from torchvision.transforms import TrivialAugmentWide

import albumentations as A
from albumentations.pytorch import ToTensorV2

from hierarchial_representation_extraction import TaxonomyParser

import imgproc

__all__ = [
    "ImageDataset",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"


class ImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, dataset_list_dir: str, image_size: int, mean: list, std: list, mode: str, taxoParser: TaxonomyParser, use_albumentation = False, maintain_aspect_ratio = False) -> None:
        super(ImageDataset, self).__init__()
        # Iterate over all image paths
        self.image_file_paths = []
        self.taxoParserObj = taxoParser
        self.use_albumentation = use_albumentation
        self.maintain_aspect_ratio = maintain_aspect_ratio

        dataset_list = os.path.join(dataset_list_dir, mode.lower() + ".txt")

        with open(dataset_list, "r") as img_list_file:
            imgList = img_list_file.readlines()

        for imgName in imgList:
            self.image_file_paths.append(os.path.join(image_dir , imgName.split('\n')[0]))
            
        # Form image class label pairs by the folder where the image is located
        self.image_size = image_size
        self.mode = mode
        self.delimiter = '/'

        if self.mode == "Train":
            if self.use_albumentation:
                # Albumentation Image pre proc
                if self.maintain_aspect_ratio:
                    self.pre_transform = A.Compose([
                                            A.LongestMaxSize(max_size=self.image_size),
                                            A.PadIfNeeded(self.image_size, self.image_size, border_mode = cv2.BORDER_CONSTANT, value=0),
                                            A.Rotate(limit=[0,270]),
                                            A.HorizontalFlip(0.5),
                                            A.VerticalFlip(0.5),
                                        ])
                else:
                    self.pre_transform = A.Compose([
                                            A.Resize((self.image_size, self.image_size)),
                                            A.Rotate(limit=[0,270]),
                                            A.HorizontalFlip(0.5),
                                            A.VerticalFlip(0.5),
                                        ])
            else:                                   
                # Use PyTorch's own data enhancement to enlarge and enhance data
                self.pre_transform = transforms.Compose([
                                        transforms.Resize((self.image_size, self.image_size)),
                                        transforms.RandomRotation([0, 270]),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.5),
                                        transforms.ColorJitter(
                                                                contrast=(0.75, 1.25),
                                                                brightness=(0.75, 1.25),
                                                                saturation=(0.75, 1.25),
                                                                hue=(-0.1, 0.1),
                                                            ),
                                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                    ])

        elif self.mode == "Valid" or self.mode == "Test":
            if self.use_albumentation:
                #Albumentation Image pre proc
                if self.maintain_aspect_ratio:
                    self.pre_transform = A.Compose([
                                        A.LongestMaxSize(max_size=self.image_size),
                                        A.PadIfNeeded(self.image_size, self.image_size, border_mode = cv2.BORDER_CONSTANT, value=0),
                                    ])
                else:
                    self.pre_transform = A.Compose([
                                        A.Resize((self.image_size, self.image_size)),])
            else:
                # Use PyTorch's own data enhancement to enlarge and enhance data
                self.pre_transform = transforms.Compose([
                                        transforms.Resize((self.image_size, self.image_size)),
                                    ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"
        if self.use_albumentation:
            #Albumentation Image post proc
            self.post_transform = A.Compose([
                                    A.Normalize( mean = mean, std = std),
                                    ToTensorV2(),
                                ])
        else:
            self.post_transform = transforms.Compose([
                                    transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize(mean, std)
                                ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int, int]:
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]

        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:
            image = cv2.imread(self.image_file_paths[batch_index])
            product = image_dir
            target = self.taxoParserObj.get_class_id(product)
            _, parent_target = self.taxoParserObj.get_class_parent(product)
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Data preprocess
        if self.use_albumentation:
            augmented = self.pre_transform(image=image)
            image = augmented['image']
        else:
            if self.maintain_aspect_ratio:
                # Data preprocess to resize to self.image_size maintaining aspect ratio
                old_size = image.shape[:2] # old_size is in (height, width) format
                ratio = float(self.image_size)/max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])

                # new_size should be in (width, height) format
                image = cv2.resize(image, (new_size[1], new_size[0])) 

                #if row/columns less than self.image_size, then pad with 0
                delta_w = self.image_size - new_size[1]
                delta_h = self.image_size - new_size[0]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)

                color = [0, 0, 0]
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                    value=color)
            # OpenCV convert PIL
            image = Image.fromarray(image)
            image = self.pre_transform(image)

        # Data postprocess
        if self.use_albumentation:
            transformed = self.post_transform(image=image)
            tensor = transformed['image']
        else:
            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            tensor = imgproc.image_to_tensor(image, False, False)
            tensor = self.post_transform(tensor)
            

        return (tensor, int(target), int(parent_target))

    def __len__(self) -> int:
        return len(self.image_file_paths)