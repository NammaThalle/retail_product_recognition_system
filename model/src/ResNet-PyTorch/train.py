import os
import gc
import time
import yaml
import json
import model
import random
import torch
import mlflow
import numpy as np

from torch import nn
from torch import optim
from dataset import ImageDataset
from argparse import ArgumentParser
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from hierarchial_representation_extraction import TaxonomyParser

from torch.backends import cudnn
from sklearn.metrics import precision_recall_fscore_support
from utils import accuracy_hierarchical, load_state_dict, save_checkpoint, Summary, AverageMeter, ProgressMeter

mlflow.autolog()
# app.run(port=5000)
    
# parse the user arguements - Added CSI
parser = ArgumentParser()
parser.add_argument('--dataset-dir', required=False, help="Enter the location of the directory train, val and test splits")
parser.add_argument('--model-data-dir', required=False, help="Enter the location of the directory train, val and test splits")
parser.add_argument('--pretrained-weights', default='', help="Enter the location of the directory where the pretrained weights are saved")
parser.add_argument('--early-stopping-patience', type=int, default=10, help="Enter the patience for early stopping")

# Model normalization parameters
parser.add_argument('--epochs', type=int, default=None, help="Enter the number of epochs to train for")
parser.add_argument('--image-size', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--num-workers', type=int, default=None)

# Loss parameters
parser.add_argument('--loss-label-smoothing', type=float, default=None)
parser.add_argument('--loss-weights', type=float, default=None)

# Optimizer parameter
parser.add_argument('--model-lr', default=None)
parser.add_argument('--model-momentum', default=None)
parser.add_argument('--model-weight-decay', default=None)
parser.add_argument('--model-ema-decay', default=None)

# Learning rate scheduler parameter
parser.add_argument('--lr-scheduler-T-0', default=None)
parser.add_argument('--lr-scheduler-T-mult', default=None)
parser.add_argument('--lr-scheduler-eta-min', default=None)

# Print frequency of results
parser.add_argument('--print-frequency', default=None)

# Model resume parameters
parser.add_argument('--resume', action='store_true', help="Add this argument for resuming training")
parser.add_argument('--resume-model-weights', help="Enter the location of the directory where the weights for resuming model training are saved")

parser.add_argument('--use-albumentation', default=None)
parser.add_argument('--maintain_image_aspect_ratio', default=None)

args = parser.parse_args()

with open(os.path.join(args.model_data_dir, 'model.json'), 'r') as trigger:
    args.model_date = json.load(trigger)['latest']

training_data_dir = os.path.join(args.model_data_dir, args.model_date, 'model_inputs')
train_image_dir = args.dataset_dir
valid_image_dir = args.dataset_dir
save_results_dir = os.path.join(args.model_data_dir, args.model_date, 'model_outputs')
hierarchy_json_file = os.path.join(training_data_dir, 'hierarchical_representation.json')

os.makedirs(save_results_dir, exist_ok=True)

if os.path.exists(os.path.join(save_results_dir, 'last.pth.tar')):
    args.resume = True
    args.resume_model_weights = os.path.join(save_results_dir, 'last.pth.tar')

# Load the YAML file
with open(os.path.join(training_data_dir, 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

config['epochs'] = args.epochs if args.epochs is not None else config['epochs']
config['image_size'] = args.image_size if args.image_size is not None else config['image_size']
config['batch_size'] = args.batch_size if args.batch_size is not None else config['batch_size']
config['num_workers'] = args.num_workers if args.num_workers is not None else config['num_workers']
config['loss_label_smoothing'] = args.loss_label_smoothing if args.loss_label_smoothing is not None else config['loss_label_smoothing']
config['loss_weights'] = args.loss_weights if args.loss_weights is not None else config['loss_weights']
config['model_lr'] = args.model_lr if args.model_lr is not None else config['model_lr']
config['model_momentum'] = args.model_momentum if args.model_momentum is not None else config['model_momentum']
config['model_weight_decay'] = args.model_weight_decay if args.model_weight_decay is not None else config['model_weight_decay']
config['model_ema_decay'] = args.model_ema_decay if args.model_ema_decay is not None else config['model_ema_decay']
config['lr_scheduler_T_0'] = args.lr_scheduler_T_0 if args.lr_scheduler_T_0 is not None else config['lr_scheduler_T_0']
config['lr_scheduler_T_mult'] = args.lr_scheduler_T_mult if args.lr_scheduler_T_mult is not None else config['lr_scheduler_T_mult']
config['lr_scheduler_eta_min'] = args.lr_scheduler_eta_min if args.lr_scheduler_eta_min is not None else config['lr_scheduler_eta_min']
config['print_frequency'] = args.print_frequency if args.print_frequency is not None else config['print_frequency']
config['maintain_image_aspect_ratio'] = args.maintain_image_aspect_ratio if args.maintain_image_aspect_ratio is not None else config['maintain_image_aspect_ratio']
config['use_albumentation'] = args.use_albumentation if args.use_albumentation is not None else config['use_albumentation']

args.pretrained_weights = os.path.join(args.model_data_dir, 'initial_weights', 'resnet50_weights.pth.tar')
print(f"\nUsing weights: {args.pretrained_weights}")

model_names = sorted(name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

# Use GPU for training by default
device = torch.device("cuda", 0)

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

config['lr_scheduler_T_0'] = config['epochs'] // 3 # Step size for LR

print("")
print(args)
print("")

def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    # Initialize json parser for hierchical data
    taxoParser = TaxonomyParser(hierarchy_json_file)
    idx_to_product = {}
    idx_to_category = {}
    
    valid_product_mask = torch.zeros((config['num_h0classes'] * config['num_h1classes']), device=device)

    for leaf in taxoParser.get_leaves():
        category_name, category_id = taxoParser.get_class_parent(leaf.name)
        product_id = (int(category_id) * config['num_h1classes'] + int(leaf.id))
        product_name = leaf.name
        if product_id not in idx_to_product:
            idx_to_product[product_id] = product_name
            valid_product_mask[product_id] = 1
        if int(category_id) not in idx_to_category:
            idx_to_category[int(category_id)] = category_name

    # Load the dataset
    _, train_loader, _, valid_loader = load_dataset(taxoParser)
    print(f"Load `{config['model_arch_name']}` datasets successfully.")

    resnet_model = build_model()
    print(f"Build `{config['model_arch_name']}` model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(resnet_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if args.pretrained_weights and not args.resume:
        resnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
            resnet_model,
            args.pretrained_weights,
            start_epoch,
            best_acc1,
            optimizer,
            scheduler)
        print(f"Loaded `{args.pretrained_weights}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if args.resume:
        resnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
            resnet_model,
            args.resume_model_weights,
            start_epoch,
            best_acc1,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py"), os.path.join(training_data_dir, "model.py"))

    # Write the dictionary to a YAML file
    with open(os.path.join(training_data_dir, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

    # Create training process log file
    writer = SummaryWriter(os.path.join(save_results_dir, "logs"))

    # Variables for early stopping
    early_stopping_counter = 0
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, config['epochs']):
            train(resnet_model, train_loader, pixel_criterion, optimizer, epoch, writer)
            acc1 = validate(resnet_model, valid_loader, pixel_criterion, epoch, writer, "Valid", valid_product_mask, idx_to_category, idx_to_product)
            
            # Log LR
            try:
                epoch_metrics = {}
                epoch_metrics['LR'] = scheduler.get_last_lr()[0]
                mlflow.log_metrics(epoch_metrics)
            except Exception as e:
                print(e)
                
            # Update LR
            scheduler.step()

            # Automatically save the model with the highest index
            is_best = acc1 > best_acc1
            is_last = (epoch + 1) == config['epochs']
            
            # Check if there has been any progress since last epoch
            if acc1 <= best_acc1:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint({"epoch": epoch + 1,
                             "best_acc1": best_acc1,
                             "state_dict": resnet_model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict()},
                            f"epoch_{epoch + 1}.pth.tar",
                            save_results_dir,
                            save_results_dir,
                            is_best,
                            is_last)
            
            # Delete older checkpoint files
            if os.path.exists(os.path.join(save_results_dir, f"epoch_{epoch - 1}.pth.tar")):
                os.remove(os.path.join(save_results_dir, f"epoch_{epoch - 1}.pth.tar"))
                
            #log best top1 val accuracy
            epoch_metrics = {}
            epoch_metrics['val/best-top1-accuracy'] = best_acc1
            mlflow.log_metrics(epoch_metrics)
            
            if epoch > 15 and early_stopping_counter >= args.early_stopping_patience:
                print(f"Validation accuracy did not improve for {args.early_stopping_patience} epochs. Stopping early at {epoch + 1} epoch")
                break
          
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

def load_dataset(taxoParser: TaxonomyParser) -> [ImageDataset, DataLoader, ImageDataset, DataLoader]:
    # Load train, test and valid datasets
    train_dataset = ImageDataset(train_image_dir,
                                 training_data_dir,
                                 config['image_size'],
                                 config['model_mean_parameters'],
                                 config['model_std_parameters'],
                                 "Train",
                                 taxoParser,
                                 config['use_albumentation'],
                                 config['maintain_image_aspect_ratio'])
								 
    valid_dataset = ImageDataset(valid_image_dir,
                                 training_data_dir,
                                 config['image_size'],
                                 config['model_mean_parameters'],
                                 config['model_std_parameters'],
                                 "Valid",
                                 taxoParser,
                                 config['use_albumentation'],
                                 config['maintain_image_aspect_ratio'])

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    return train_dataset, train_dataloader, valid_dataset, valid_dataloader

def build_model() -> nn.Module:
    resnet_model = model.__dict__[config['model_arch_name']](num_classes_h0 = config['num_h0classes'], num_classes_h1 = config['num_h1classes'])
    resnet_model.to(device=device, memory_format=torch.channels_last)
    return resnet_model

def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config['loss_label_smoothing'])
    #criterion = nn.NLLLoss()
    criterion = criterion.to(device=device, memory_format=torch.channels_last)
    return criterion

def define_optimizer(model):
    optimizer = optim.Adam(model.parameters(),
                      lr=config['model_lr'],
                      weight_decay=config['model_weight_decay'])
    return optimizer

def define_scheduler(optimizer: optim.SGD):
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size = config['lr_scheduler_T_0'],
                                    gamma = 0.1)
    return scheduler

def train(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        epoch: int,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_loader)
    
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    acc1_h0 = AverageMeter("Acc@1-category", ":6.2f")
    acc5_h0 = AverageMeter("Acc@5-category", ":6.2f")
    acc1_h1 = AverageMeter("Acc@1-product", ":6.2f")
    acc5_h1 = AverageMeter("Acc@5-product", ":6.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1_h0, acc5_h0, acc1_h1, acc5_h1],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Get the initialization training time
    end = time.time()

    for i, (images, target, parent_target) in enumerate(train_loader):
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = images.to(device=device, memory_format=torch.channels_last, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        parent_target = parent_target.to(device=device, non_blocking=True)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Non-mixed precision training
        probs = model(images, target, parent_target)
		
        loss = -torch.mean(torch.log(probs))
        #loss = criterion(torch.log(probs), parent_target)

        # Backpropagation
        loss.backward()
        
        # Update generator weights
        optimizer.step()

        # Measure accuracy and record loss
        class_probs = model(images)
        acc_h0, acc_h1 = accuracy_hierarchical(class_probs, target, parent_target, topk=(1, 5))
        #top1_h0, top5_h0 = accuracy(class_probs, parent_target, topk=(1, 2))

        top1_h0, top5_h0 = acc_h0
        top1_h1, top5_h1 = acc_h1
        losses.update(loss.item(), batch_size)
        acc1_h0.update(top1_h0[0].item(), batch_size)
        acc5_h0.update(top5_h0[0].item(), batch_size)
        acc1_h1.update(top1_h1[0].item(), batch_size)
        acc5_h1.update(top5_h1[0].item(), batch_size)
		
        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config['print_frequency'] == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

        del images, target, parent_target, probs, class_probs
        torch.cuda.empty_cache()
        gc.collect()
    
    epoch_metrics = {}
    epoch_metrics['train/loss'] = losses.avg
    epoch_metrics['train/category-top1-accuracy'] = acc1_h0.avg
    epoch_metrics['train/category-top5-accuracy'] = acc5_h0.avg
    epoch_metrics['train/product-top1-accuracy'] = acc1_h1.avg
    epoch_metrics['train/product-top5-accuracy'] = acc5_h1.avg
    mlflow.log_metrics(epoch_metrics)

def validate(
        model: nn.Module,
        valid_loader: DataLoader,
        criterion: nn.CrossEntropyLoss, 
        epoch: int,
        writer: SummaryWriter,
        mode: str,
        valid_product_mask: torch.tensor,
        idx_to_category: dict, 
        idx_to_product: dict
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(valid_loader)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f")
    acc1_h0 = AverageMeter("Acc@1-category", ":6.2f", Summary.AVERAGE)
    acc5_h0 = AverageMeter("Acc@5-category", ":6.2f", Summary.AVERAGE)
    acc1_h1 = AverageMeter("Acc@1-product", ":6.2f", Summary.AVERAGE)
    acc5_h1 = AverageMeter("Acc@5-product", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, losses, acc1_h0, acc5_h0, acc1_h1, acc5_h1], prefix=f"{mode}: ")

    # Put the exponential moving average model in the verification mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Get the initialization test time
    end = time.time()

    pred_h0_list=torch.zeros(0,dtype=torch.long, device=device)
    pred_h1_list=torch.zeros(0,dtype=torch.long, device=device)
    target_h0_list=torch.zeros(0,dtype=torch.long, device=device)
    target_h1_list=torch.zeros(0,dtype=torch.long, device=device)

    epoch_metrics = {}
    
    with torch.no_grad():
        for images, target, parent_target in valid_loader:
            # Transfer in-memory data to CUDA devices to speed up training
            images = images.to(device=device, memory_format=torch.channels_last, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            parent_target = parent_target.to(device=device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            probs = model(images, target, parent_target)

            # measure accuracy and record loss
            loss = -torch.mean(torch.log(probs))
            #loss = criterion(torch.log(probs), parent_target)
            losses.update(loss.item(), batch_size)

            class_probs = model(images)
            acc_h0, acc_h1 = accuracy_hierarchical(class_probs, target, parent_target, topk=(1, 5))
            #top1_h0, top5_h0 = accuracy(class_probs, parent_target, topk=(1, 2))

            top1_h0, top5_h0 = acc_h0
            top1_h1, top5_h1 = acc_h1
            acc1_h0.update(top1_h0[0].item(), batch_size)
            acc5_h0.update(top5_h0[0].item(), batch_size)
            acc1_h1.update(top1_h1[0].item(), batch_size)
            acc5_h1.update(top5_h1[0].item(), batch_size)

            #print(loss.item())
            
            flat_class_probs = class_probs.flatten(start_dim=-2)
            # _, pred_h1_test = torch.max(flat_class_probs, 1)
            flat_class_probs_filtered = torch.mul(valid_product_mask, flat_class_probs)
            _, pred_h1 = torch.max(flat_class_probs_filtered, 1)

            pred_h0 = torch.div(pred_h1, class_probs.size(-1), rounding_mode='trunc')
            
            target = parent_target * class_probs.size(-1) + target

            # Append category prediction results
            pred_h0_list=torch.cat([pred_h0_list,pred_h0.view(-1)])
            target_h0_list=torch.cat([target_h0_list,parent_target.view(-1)])

            # Append product prediction results
            pred_h1_list=torch.cat([pred_h1_list,pred_h1.view(-1)])
            target_h1_list=torch.cat([target_h1_list,target.view(-1)])
            
            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config['print_frequency'] == 0:
                progress.display(batch_index + 1)

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

            del images, target, parent_target, probs, class_probs
            torch.cuda.empty_cache()
            gc.collect()

        category_index_list = set()
        for pred in pred_h0_list.cpu().numpy():
            category_index_list.add(pred)
        for target in target_h0_list.cpu().numpy():
            category_index_list.add(target)

        # product wise lists for results.json
        product_index_list = set()
        for pred in pred_h1_list.cpu().numpy():
            product_index_list.add(pred)
        for target in target_h1_list.cpu().numpy():
            product_index_list.add(target)
        
        # Category list for CF
        category_list = []
        for idx in sorted(category_index_list):
            category_list.append(idx_to_category[idx])
            
        # product list for CF
        product_list = []
        for idx in sorted(product_index_list):
            product_list.append(idx_to_product[idx])
            
        metrics_report_category = precision_recall_fscore_support(target_h0_list.cpu().numpy(), pred_h0_list.cpu().numpy(), average='weighted', zero_division=np.nan)
        metrics_report_product = precision_recall_fscore_support(target_h1_list.cpu().numpy(), pred_h1_list.cpu().numpy(), average='weighted', zero_division=np.nan)
        
        epoch_metrics['val/category-weighted-precision'] = metrics_report_category[0]
        epoch_metrics['val/category-weighted-recall'] = metrics_report_category[1]
        epoch_metrics['val/category-weighted-f1-score'] = metrics_report_category[2]
        epoch_metrics['val/product-weighted-precision'] = metrics_report_product[0] 
        epoch_metrics['val/product-weighted-recall'] = metrics_report_product[1] 
        epoch_metrics['val/product-weighted-f1-score'] = metrics_report_product[2]
        
    # print metrics
    progress.display_summary()

    epoch_metrics['val/loss'] = losses.avg
    epoch_metrics['val/category-top1-accuracy'] = acc1_h0.avg
    epoch_metrics['val/category-top5-accuracy'] = acc5_h0.avg
    epoch_metrics['val/product-top1-accuracy'] = acc1_h1.avg
    epoch_metrics['val/product-top5-accuracy'] = acc5_h1.avg
    mlflow.log_metrics(epoch_metrics)

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/loss", losses.avg, epoch + 1)
        writer.add_scalar(f"{mode}/category-Acc@1", acc1_h0.avg, epoch + 1)
        writer.add_scalar(f"{mode}/category-Acc@5", acc5_h0.avg, epoch + 1)
        writer.add_scalar(f"{mode}/product-Acc@1", acc1_h1.avg, epoch + 1)
        writer.add_scalar(f"{mode}/product-Acc@5", acc5_h1.avg, epoch + 1)
        writer.add_scalar(f"{mode}/category-weighted-precision", metrics_report_category[0], epoch + 1)
        writer.add_scalar(f"{mode}/category-weighted-recall", metrics_report_category[1], epoch + 1)
        writer.add_scalar(f"{mode}/category-weighted-f1-score", metrics_report_category[2], epoch + 1)
        writer.add_scalar(f"{mode}/product-weighted-precision", metrics_report_product[0], epoch + 1)
        writer.add_scalar(f"{mode}/product-weighted-recall", metrics_report_product[1], epoch + 1)
        writer.add_scalar(f"{mode}/product-weighted-f1-score", metrics_report_product[2], epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1_h1.avg

if __name__ == "__main__":
    main()
