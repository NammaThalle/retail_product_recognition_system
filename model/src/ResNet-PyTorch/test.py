
import os
import cv2
import time
import json
import math
import yaml
import torch
import mlflow
import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from torch import nn
from dataset import ImageDataset
from torch.backends import cudnn
from torch.utils.data import DataLoader
from importlib.machinery import SourceFileLoader
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from hierarchial_representation_extraction import TaxonomyParser
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from utils import load_state_dict, accuracy_hierarchical, Summary, AverageMeter, ProgressMeter

mlflow.autolog()
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# parse the user arguements - Added CSI
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dataset-dir', required=True, help="Enter the location of the directory train, val and test splits")
parser.add_argument('--model-data-dir', required=True, help="Enter the location of the directory train, val and test splits")
args = parser.parse_args()

# Use GPU for training by default
device = torch.device("cuda", 0)

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

with open(os.path.join(args.model_data_dir, 'model.json'), 'r') as trigger:
    args.model_date = json.load(trigger)['latest']

test_image_dir = args.dataset_dir
save_results_dir = os.path.join(args.model_data_dir, args.model_date, 'model_outputs')
model_weights_path = os.path.join(save_results_dir, "best.pth.tar")
training_data_dir = os.path.join(args.model_data_dir, args.model_date, 'model_inputs')
hierarchy_json_file = os.path.join(training_data_dir, 'hierarchical_representation.json')

# Load the YAML file
with open(os.path.join(training_data_dir, 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

query_image_save = os.path.join(save_results_dir, "confusion_analysis")
os.makedirs(query_image_save, exist_ok=True)

model_file_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py")
model = SourceFileLoader("model", model_file_location).load_module()

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

def build_model() -> nn.Module:
    resnet_model = model.__dict__[config['model_arch_name']](num_classes_h0 = config['num_h0classes'], num_classes_h1 = config['num_h1classes'])
    resnet_model = resnet_model.to(device=device, memory_format=torch.channels_last)

    return resnet_model

def load_dataset(taxoParser: TaxonomyParser) -> [ImageDataset, DataLoader]:
    test_dataset = ImageDataset(test_image_dir,
                                training_data_dir,
                                config['image_size'],
                                config['model_mean_parameters'],
                                config['model_std_parameters'],
                                "Test",
                                taxoParser,
                                config['use_albumentation'],
                                config['maintain_image_aspect_ratio'])
								
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    return test_dataset, test_dataloader

def create_json_result(y_true, y_pred, target_names, cm, acc_category=None, pred_category=None):

    print("target_names: ", len(target_names)) #," | acc_category: ", len(acc_category))
    # Calculate overall accuracy
    total_correct = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]])
    accuracy = (total_correct / len(y_true)) * 100
    
    # Create dictionary for result
    result_dict = {target_names[i]: {'accuracy': 0, 'category': None, 'miss_classified': {}} for i in range(len(target_names))}

    # Populate dictionary with accuracy and miss-classified percentages
    for label_idx, label in enumerate(target_names):
        # result_dict[label]['category'] = acc_category[label_idx]
        result_dict[label]['accuracy'] = (cm[label_idx][label_idx]) * 100
        if result_dict[label]['accuracy'] < 99:
             for other_class_idx, other_class in enumerate(cm[label_idx]):
                if other_class_idx == label_idx or (other_class * 100) <= 1:
                     continue
                result_dict[label]['miss_classified'][target_names[other_class_idx]] = other_class * 100
                
        if len(result_dict[label]['miss_classified']) == 0:
             del result_dict[label]['miss_classified']
    
    # Sort the dictionary items by the "accuracy" value in each nested dictionary
    sorted_dict = dict(sorted(result_dict.items(), key=lambda x: x[1]["accuracy"], reverse=True))

    # Add overall accuracy to result dictionary
    sorted_dict['overall_accuracy'] = accuracy

    # Convert result dictionary to JSON and return it
    return sorted_dict

def create_image_collage_no_resize(main_image, main_product, main_percent, main_category, miss_image_list=None, miss_product_list=None, miss_percent_list=None):
    query_h, query_w, _ = main_image.shape

    # Padding between images
    padding = 10
    label_padding = 5

    # Maximum width of label
    max_label_width = 120

    # Calculate maximum dimensions of result images
    max_result_h = max(result.shape[0] for result in miss_image_list)
    max_result_w = max(result.shape[1] for result in miss_image_list)

    # Calculate number of rows and columns in collage
    num_cols = min(len(miss_image_list), 5)
    num_rows = int(math.ceil(len(miss_image_list) / num_cols)) + 1

    # Calculate size of collage
    collage_w = query_w + (num_cols * max_result_w) + ((num_cols - 1) * padding) + 500
    collage_h = query_h + (num_rows * max_result_h) + ((num_rows - 1) * padding) + label_padding
    collage = np.full((collage_h-200, collage_w, 3), 255, dtype=np.uint8)

    # Add query image to collage
    query_pos = (int((collage_w - query_w) / 2), label_padding)
    collage[query_pos[1]:query_pos[1]+query_h, query_pos[0]:query_pos[0]+query_w] = main_image

    # Add result images to collage
    for i, result in enumerate(miss_image_list):
        
        col_pad = (i%5)*100
        row_pad = 70 if i < 5 else 100

        # Compute position of result image in collage
        row_pos = query_h + (i // num_cols) * (max_result_h + padding) + row_pad 
        col_pos = query_w + (i % num_cols) * (max_result_w + padding) + col_pad

        # Add result image to collage
        result_h, result_w, _ = result.shape
        collage[row_pos:row_pos+result_h, col_pos:col_pos+result_w] = result

        # Add label to result image
        label = f'{miss_product_list[i]}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_size = (min(label_size[0], max_label_width), label_size[1])
        label_pos = (col_pos + int((result_w - label_size[0]) / 2), row_pos + result_h + label_size[1]+ 10)
        color = (0, 0, 0)
        if miss_product_list[i] != main_product:
            color = (0, 0, 255)
        cv2.putText(collage, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        label_pos = (col_pos + int((result_w - label_size[0]) / 2), row_pos + result_h + label_size[1]+ 30)
        cv2.putText(collage, f' Accuracy: {round(miss_percent_list[i],3)}', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Add label to main image
    label = f'{main_product} | {main_category} | Accuracy:{round(main_percent,3)}'
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)
    label_size = (min(label_size[0], max_label_width), label_size[1])
    label_pos = (query_pos[0] + int((query_w - label_size[0]) / 2), query_pos[1] + query_h + label_size[1] + padding)
    label_pos = (max(0, label_pos[0]-250), min(label_pos[1], collage_h - label_size[1]))
    cv2.putText(collage, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the collage
    product_save_path = os.path.join(query_image_save)
    os.makedirs(product_save_path, exist_ok=True)
    cv2.imwrite(os.path.join(product_save_path, f'{round(main_percent)}_{main_product}.jpg'), collage)

def main() -> None:
    # Initialize the model
    resnet_model = build_model()
    print(f"Build `{config['model_arch_name']}` model successfully.")

    # Load model weights
    resnet_model, _, _, _, _ = load_state_dict(resnet_model, model_weights_path)
    print(f"Load `{config['model_arch_name']}` "
          f"model weights `{os.path.abspath(model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # Initialize json parser for hierchical data
    taxoParser = TaxonomyParser(hierarchy_json_file)
    idx_to_product = {}
    idx_to_category = {}

    valid_products_mask = torch.zeros((config['num_h0classes'] * config['num_h1classes']),  device=device)

    for leaf in taxoParser.get_leaves():
        category_name, category_id = taxoParser.get_class_parent(leaf.name)
        product_id = (int(category_id) * config['num_h1classes'] + int(leaf.id))
        product_name = leaf.name
        if product_id not in idx_to_product:
            idx_to_product[product_id] = product_name
            valid_products_mask[product_id] = 1
        if int(category_id) not in idx_to_category:
            idx_to_category[int(category_id)] = category_name

    # Load test dataloader
    _, test_loader = load_dataset(taxoParser)

    # Calculate how many batches of data are in each Epoch
    batches = len(test_loader)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1_h0 = AverageMeter("Acc@1-category", ":6.2f", Summary.AVERAGE)
    acc5_h0 = AverageMeter("Acc@5-category", ":6.2f", Summary.AVERAGE)
    acc1_h1 = AverageMeter("Acc@1-product", ":6.2f", Summary.AVERAGE)
    acc5_h1 = AverageMeter("Acc@5-product", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1_h0, acc5_h0, acc1_h1, acc5_h1], prefix=f"Test: ")

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    pred_h0_list=torch.zeros(0,dtype=torch.long, device=device)
    pred_h1_list=torch.zeros(0,dtype=torch.long, device=device)
    target_h0_list=torch.zeros(0,dtype=torch.long, device=device)
    target_h1_list=torch.zeros(0,dtype=torch.long, device=device)

    # Get the initialization test time
    end = time.time()
    
    pred_confidence = list()

    with torch.no_grad():
        for i, (images, target, parent_target) in enumerate(test_loader):
            # Transfer in-memory data to CUDA devices to speed up training
            images = images.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            parent_target = parent_target.to(device=device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = resnet_model(images)

            # measure accuracy and record loss
            acc_h0, acc_h1 = accuracy_hierarchical(output, target, parent_target, topk=(1, 5))
            top1_h0, top5_h0 = acc_h0
            top1_h1, top5_h1 = acc_h1
            acc1_h0.update(top1_h0[0].item(), batch_size)
            acc5_h0.update(top5_h0[0].item(), batch_size)
            acc1_h1.update(top1_h1[0].item(), batch_size)
            acc5_h1.update(top5_h1[0].item(), batch_size)

            flat_class_probs = output.flatten(start_dim=-2)
            # _, pred_h1_test = torch.max(flat_class_probs, 1)
            flat_class_probs_filtered = torch.mul(valid_products_mask, flat_class_probs)
            _, pred_h1 = torch.max(flat_class_probs_filtered, 1)
            
            pred_h0 = torch.div(pred_h1, output.size(-1), rounding_mode='trunc')

            target = parent_target * output.size(-1) + target

            # Append category prediction results
            pred_h0_list=torch.cat([pred_h0_list,pred_h0.view(-1)])
            target_h0_list=torch.cat([target_h0_list,parent_target.view(-1)])

            # Append product prediction results
            pred_h1_list=torch.cat([pred_h1_list,pred_h1.view(-1)])
            target_h1_list=torch.cat([target_h1_list,target.view(-1)])
            
            # Calculate Prediction Confidence
            for batch_idx in range(images.size(0)):
                category_index = pred_h0[batch_idx].cpu().item()
                product_index = pred_h1[batch_idx].cpu().item()
                product_index = product_index % output.size(-1)
                prediction_product_prob = output[batch_idx, category_index, product_index].item()
                pred_confidence.append(prediction_product_prob)
                
            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config['print_frequency'] == 0:
                progress.display(batch_index + 1)

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    print(f"top1 category error: {100 - acc1_h0.avg:.2f}%")
    print(f"top5 category error: {100 - acc5_h0.avg:.2f}%")
    print(f"top1 product error: {100 - acc1_h1.avg:.2f}%")
    print(f"top5 product error: {100 - acc5_h1.avg:.2f}%")

    print(f"Acc@1 category: {acc1_h0.avg}")
    print(f"Acc@5 category: {acc5_h0.avg}")
    print(f"Acc@1 product: {acc1_h1.avg}")
    print(f"Acc@5 product: {acc5_h1.avg}")

    print('Generating results.json')
    # Category wise lists for results.json
    pred_category_list = []
    target_category_list = []
    category_index_list = set()
    for pred in pred_h0_list.cpu().numpy():
        category_index_list.add(pred)
        pred_category_list.append(idx_to_category[pred])
    for target in target_h0_list.cpu().numpy():
        category_index_list.add(target)
        target_category_list.append(idx_to_category[target])

    # product wise lists for results.json
    pred_product_list = []
    target_product_list = []
    product_index_list = set()
    for pred in pred_h1_list.cpu().numpy():
        product_index_list.add(pred)
        pred_product_list.append(idx_to_product[pred])
    for target in target_h1_list.cpu().numpy():
        product_index_list.add(target)
        target_product_list.append(idx_to_product[target])
    
    # Create results.json data
    jsonData = {
        "results":[]
        }
    for idx in range(len(pred_product_list)):
        detectionDict = {}
        detectionDict["Category Actual"] = target_category_list[idx]
        detectionDict["Category Prediction"] = pred_category_list[idx]
        detectionDict["Product Actual"] = target_product_list[idx]
        detectionDict["Product Prediction"] = pred_product_list[idx]
        detectionDict["Product Confidence"] = pred_confidence[idx]
        jsonData["results"].append(detectionDict)
    
    with open(os.path.join(save_results_dir, "results.json"),"w") as jsonFile:
        json.dump(jsonData, jsonFile, indent = 4)
    
    # Category list for CF
    category_list = []
    for idx in sorted(category_index_list):
        category_list.append(idx_to_category[idx])
        
    # product list for CF
    product_list = []
    for idx in sorted(product_index_list):
        product_list.append(idx_to_product[idx])

    print('Generating Confusion Matrix - Category')
    # Category wise CF
    confusion_matrix_category = confusion_matrix(target_h0_list.cpu().numpy(), pred_h0_list.cpu().numpy())
    df_cm = pd.DataFrame(confusion_matrix_category, index = category_list, columns = category_list)
    
    plt.figure(figsize = (10, 10))
    s = sn.heatmap(df_cm, annot=True, fmt='d')
    plt.title('Confusion Matrix for Categorical Classification', fontsize = 20) # title with fontsize 20
    plt.xlabel('Predictions', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Actuals', fontsize = 15) # y-axis label with fontsize 15
    plotImgFileName = os.path.join(save_results_dir, "CF_category.jpg")
    plt.savefig(plotImgFileName, bbox_inches='tight')
    
    print('Generating Confusion Matrix - Product')
    # product wise CF
    confusion_matrix_product = confusion_matrix(target_h1_list.cpu().numpy(), pred_h1_list.cpu().numpy())
    df_cm = pd.DataFrame(confusion_matrix_product, index = product_list, columns = product_list)
    df_cm = df_cm.sort_index(axis=1).sort_index()
    df_cm.to_csv(os.path.join(save_results_dir, 'confusion_matrix.csv'))
    
    plt.figure(figsize = (200, 200))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.title('Confusion Matrix for Product Classification', fontsize = 20) # title with fontsize 20
    plt.xlabel('Predictions', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Actuals', fontsize = 15) # y-axis label with fontsize 15
    plotImgFileName = os.path.join(save_results_dir, "CF_product.jpg")
    plt.savefig(plotImgFileName, bbox_inches='tight')
    
    print('Generating Confusion Matrix - Product - Normalized')
    # product wise normalized CF
    # confusion_matrix_product_norm = confusion_matrix(target_h1_list.cpu().numpy(), pred_h1_list.cpu().numpy(), normalize='true')
    # df_cm = pd.DataFrame(confusion_matrix_product_norm, index = product_list, columns = product_list)
    # df_cm = df_cm.sort_index(axis=1).sort_index()
    # csv_file_name = os.path.join(save_results_dir, 'CF_product_normalized.csv')
    # df_cm.to_csv(csv_file_name)
    
    # plt.figure(figsize = (200, 200))
    # s = sn.heatmap(df_cm, annot=True, fmt='.2f', )
    # s.set_xticklabels(s.get_xticklabels(), rotation=90)
    # s.set_yticklabels(s.get_yticklabels(), rotation=0)
    # plt.title('Normalized Confusion Matrix for Product Classification', fontsize = 20) # title with fontsize 20
    # plt.xlabel('Predictions', fontsize = 15) # x-axis label with fontsize 15
    # plt.ylabel('Actuals', fontsize = 15) # y-axis label with fontsize 15
    # plotImgFileName = os.path.join(save_results_dir, "CF_product_normalized.jpg")
    # plt.savefig(plotImgFileName)
    
    print('Generating Metrics - Category')
    # Classification Report for Category (Precision, Recall, F1-Score)
    try:
        classification_report_category = classification_report(target_h0_list.cpu().numpy(), pred_h0_list.cpu().numpy(), target_names = category_list, output_dict = True)
        metricDF = pd.DataFrame.from_dict({i: classification_report_category[i] for i in category_list}, orient='index')
        metricDF.to_csv(os.path.join(save_results_dir, "Category_Metrics.csv"))
    except UndefinedMetricWarning:
        pass
    
    print('Generating Metrics - product')
    # Classification Report for product (Precision, Recall, F1-Score)
    try:
        classification_report_product = classification_report(target_h1_list.cpu().numpy(), pred_h1_list.cpu().numpy(), target_names = product_list, output_dict = True)
        product_list.sort()
        product_metrics_df = pd.DataFrame.from_dict({i: classification_report_product[i] for i in product_list}, orient='index')
        product_metrics_df.insert(0, 'UnivUpc', product_metrics_df.index)
        product_metrics_df.to_csv(os.path.join(save_results_dir, "product_metrics.csv"))
    except UndefinedMetricWarning:
        pass
    
    metrics_report_category = precision_recall_fscore_support(target_h0_list.cpu().numpy(), pred_h0_list.cpu().numpy(), average='weighted', zero_division=np.nan)
    metrics_report_product = precision_recall_fscore_support(target_h1_list.cpu().numpy(), pred_h1_list.cpu().numpy(), average='weighted', zero_division=np.nan)
    
    # MLFLOW logs
    metrics = {}
    metrics['test/category-top1-accuracy'] = acc1_h0.avg
    metrics['test/category-top5-accuracy'] = acc5_h0.avg
    metrics['test/product-top1-accuracy'] = acc1_h1.avg
    metrics['test/product-top5-accuracy'] = acc5_h1.avg
    metrics['test/category-weighted-precision'] = metrics_report_category[0]
    metrics['test/category-weighted-recall'] = metrics_report_category[1]
    metrics['test/category-weighted-f1-score'] = metrics_report_category[2]
    metrics['test/product-weighted-precision'] = metrics_report_product[0] 
    metrics['test/product-weighted-recall'] = metrics_report_product[1] 
    metrics['test/product-weighted-f1-score'] = metrics_report_product[2]
    mlflow.log_metrics(metrics)
    
    with open(os.path.join(save_results_dir, 'test_metrics.json'), 'w') as file:
        json.dump({"test_metrics": metrics}, file)

if __name__ == "__main__":    
    main()