import os
import shutil
import datetime
from enum import Enum
from typing import Any, Dict, TypeVar, Optional

import torch
from torch import nn

__all__ = [
    "accuracy", "accuracy_hierarchical", "load_state_dict", "make_directory", "ovewrite_named_param", "make_divisible", "compare_models", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter"
]

V = TypeVar("V")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results

def accuracy_hierarchical(class_probs, target, parent_target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        target = parent_target * class_probs.size(-1) + target

        flat_class_probs = class_probs.flatten(start_dim=-2)
        _, pred_h1 = flat_class_probs.topk(maxk, 1, True, True)
        pred_h0 = torch.div(pred_h1, class_probs.size(-1), rounding_mode='trunc')

        pred_h0 = pred_h0.t()
        pred_h1 = pred_h1.t()
        correct_h0 = pred_h0.eq(parent_target.view(1, -1).expand_as(pred_h0))
        correct_h1 = pred_h1.eq(target.view(1, -1).expand_as(pred_h1))

        #modifying correct_h0 to consider hierarchical way of getting the parent value from model
        # issue e.g. - if we have top3 values child classes and more than 1 belongs to same parent class,
        #              then the parent class values in topk won't differ and hence the number of True values
        #              in correct_h0 will increase cauing top3 accuracy more than 100%.
        # solution -   To logically OR the values so that we will consider only one of True values if available and not all
        #              True values. In case, there's no True value, the final value will be False
        
        correct_h0_final = torch.unsqueeze(correct_h0[0],0)
        for k in range(1, maxk):
            temp = correct_h0_final.logical_or(correct_h0[k])
            correct_h0_final = torch.cat((correct_h0_final, temp), dim=0)

        results_h0 = []
        results_h1 = []
        for k in topk:
            correct_k = correct_h0_final[k-1].reshape(-1).float().sum(0, keepdim=True)
            results_h0.append(correct_k.mul_(100.0 / batch_size))

            correct_k = correct_h1[:k].reshape(-1).float().sum(0, keepdim=True)
            results_h1.append(correct_k.mul_(100.0 / batch_size))

        return results_h0, results_h1


def load_state_dict(
        model: nn.Module,
        model_weights_path: str,
        start_epoch: int = None,
        best_acc1: float = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage, weights_only=True)

    if load_mode == "resume":
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

    return model, start_epoch, best_acc1, optimizer, scheduler


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Copy from: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def compare_models(model_1, model_2):
    models_differ = 0
    m1sd = model_1.state_dict()
    m2sd = model_2.state_dict()

    for key_item_1 in m1sd.keys():
        if key_item_1 in m2sd.keys():
            if torch.equal(m1sd[key_item_1], m2sd[key_item_1]):
                pass
            else:
                models_differ += 1
                print("values diff for layer :",key_item_1 )
                print(m1sd[key_item_1])
                print(m2sd[key_item_1])
        else:
            print("m1 Key not present in m2", key_item_1)

    if models_differ == 0:
        print('Models match perfectly! :)')
    else:
        print(models_differ)

def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "best.pth.tar"))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "last.pth.tar"))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: '] 
        entries += [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print('\n')
        print("=" * 50)
        print(" ".join(entries))
        print("=" * 50)
        print('\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

