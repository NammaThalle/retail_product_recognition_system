
import os
import cv2
import yaml
import torch
import random
import warnings

import numpy as np
import resnet.imgproc

from torch import nn
from PIL import Image
from resnet.utils import load_state_dict
from importlib.machinery import SourceFileLoader
from sklearn.exceptions import UndefinedMetricWarning
from resnet.hierarchial_representation_extraction import TaxonomyParser
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class ProductModel(object):
    def __init__(self):
        
        self.topK = 1
        self.resnet_model = None
        self.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        
        self.model_dir =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
        self.model_weight_path = os.path.join(self.model_dir, 'best.pth.tar')
        self.config_file_path = os.path.join(self.model_dir, 'config.yaml')
        self.hierarchical_file_path = os.path.join(self.model_dir, 'hierarchical_representation.json')
        self.model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py')
        
        try:
            # Load the YAML file
            with open(self.config_file_path, 'r') as file:
                config = yaml.safe_load(file)
            
            self.taxoParser = TaxonomyParser(self.hierarchical_file_path)
        
        except Exception as e:
            print(f"Error loading config or hierarchical representation files: {str(e)}")

        self.image_size = config['image_size']
        self.num_h0classes = config['num_h0classes']
        self.num_h1classes = config['num_h1classes']
        self.model_architecture_name = config['model_arch_name']
        self.model_mean_params = config['model_mean_parameters']
        self.model_std_params = config['model_std_parameters']
        self.idx_to_product = {}
        self.idx_to_category = {}

        self.valid_products_mask = torch.zeros((self.num_h0classes * self.num_h1classes),  device=self.device)

        for leaf in self.taxoParser.get_leaves():
            category_name, category_id = self.taxoParser.get_class_parent(leaf.name)
            product_id = (int(category_id) * self.num_h1classes + int(leaf.id))
            product_name = leaf.name
            if product_id not in self.idx_to_product:
                self.idx_to_product[product_id] = product_name
                self.valid_products_mask[product_id] = 1
            if int(category_id) not in self.idx_to_category:
                self.idx_to_category[int(category_id)] = category_name

        self.resnet_model = self.build_model()

        if self.resnet_model is None:
            raise ValueError("Failed to build model.")
        
        self.resnet_model, _, _, _, _ = load_state_dict(self.resnet_model, self.model_weight_path)
        
        if self.resnet_model is None:
            raise ValueError("Failed to load model weights.")
        
        self.resnet_model.eval()

    def build_model(self) -> nn.Module:
        model = SourceFileLoader("model", self.model_file_path).load_module()
        resnet_model = model.__dict__[self.model_architecture_name](num_classes_h0 = self.num_h0classes,  num_classes_h1 = self.num_h1classes)
        resnet_model = resnet_model.to(device=self.device, memory_format=torch.channels_last)

        return resnet_model 

    def preprocess_image(self, image):
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV convert PIL
        image = Image.fromarray(image)

        # Resize to 224
        image = Resize([self.image_size, self.image_size])(image)
        # Convert image data to pytorch format data
        tensor = resnet.imgproc.image_to_tensor(image, False, False).unsqueeze_(0)
        # Convert a tensor image to the given ``dtype`` and scale the values accordingly
        tensor = ConvertImageDtype(torch.float)(tensor)
        # Normalize a tensor image with mean and standard deviation.
        tensor = Normalize(self.model_mean_params, self.model_std_params)(tensor)

        # Transfer tensor channel image format data to CUDA device
        tensor = tensor.to(device=self.device, memory_format=torch.channels_last, non_blocking=True)

        return tensor

    def classifyProducts(self, image):

        image = self.preprocess_image(image)

        with torch.no_grad():
            image = image.to(device=self.device, non_blocking=True)
            output = self.resnet_model(image)

        self.classified_objects = []

        flat_class_probs = output.flatten(start_dim=-2)
        _, pred_h1 = flat_class_probs.topk(k = 3, dim = 1, largest = True, sorted = True)
        pred_h0 = torch.div(pred_h1, output.size(-1), rounding_mode='trunc')

        # Print classification results
        products = []
        pred_h0_list = pred_h0.tolist()[0]
        pred_h1_list = pred_h1.tolist()[0]
        for category_index, sku_index in zip(pred_h0_list, pred_h1_list):
            prediction_category_label = self.idx_to_category[category_index]
            prediction_sku_label = self.idx_to_product[sku_index]
            sku_index = sku_index%output.size(-1)
            prediction_sku_prob = output[0, category_index, sku_index].item()
            product = {}
            product['category'] = prediction_category_label
            product['name'] = prediction_sku_label
            product['confidence'] = str(round(prediction_sku_prob * 100))
            products.append(product)
        self.classified_objects.extend(products)

        return self.classified_objects