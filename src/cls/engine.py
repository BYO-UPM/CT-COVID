import os
import json
import torch
import numpy as np
from torch.nn.functional import cross_entropy
from tqdm import tqdm


    
def train_model(epoch, model, train_loader, optimizer, scheduler, train_metrics, args):
    model.train()
    train_metrics.initialize_empty_epoch_metrics(epoch, "train")
    for idx, (input, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} - train"):
        if args.cuda:
            input = input.to(args.device)
            labels = labels.to(args.device)
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        correct, total, accuracy = train_metrics.accuracy(output, labels)
        confusion_matrix = train_metrics.confusion_matrix(output, labels)
        train_metrics.update_minibatch_metrics(epoch, "train", total=total, correct=correct, accuracy=accuracy,
                                                loss=loss.item(), confusion_matrix=confusion_matrix)
    train_metrics.log_epoch_metrics(epoch, "train", total_batches=idx, print=True)
    return train_metrics.get_epoch_metrics(epoch, "train")


def validate_model(epoch, model, val_loader, val_metrics, args):
    model.eval()
    val_metrics.initialize_empty_epoch_metrics(epoch, "val")
    with torch.no_grad():
        for idx, (input, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} - val"):
            if args.cuda:
                input = input.to(device=args.device)
                labels = labels.to(device=args.device)
            output = model(input)
            loss = cross_entropy(output, labels)
            correct, total, accuracy = val_metrics.accuracy(output, labels)
            confusion_matrix = val_metrics.confusion_matrix(output, labels)
            val_metrics.update_minibatch_metrics(epoch, "val", total=total, correct=correct, accuracy=accuracy, loss=loss.item(), confusion_matrix=confusion_matrix)
        val_metrics.log_epoch_metrics(epoch, "val", total_batches=idx, print=True)

    return val_metrics.get_epoch_metrics(epoch, "val")


def save_checkpoint(state, path,filename='last'):

    name = os.path.join(path, filename+'_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)

def save_model(model, optimizer, args, metrics, epoch, best_metric, target_metric="accuracy", goal="maximize"):
    current_epoch_metrics = metrics.get_epoch_metrics(epoch, "val")
    target_metric_value = current_epoch_metrics[target_metric]
    save_path = args.save
    os.makedirs(save_path, exists_ok=True)
    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    is_best = False
    if epoch > args.save_patience:
        if goal == "maximize":
            if target_metric_value > best_metric:
                is_best = True
                best_metric = target_metric_value
                confusion_matrix = current_epoch_metrics["confusion_matrix"]
                save_checkpoint({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'metrics': current_epoch_metrics}, save_path, args.model + "_best")
                np.save(save_path + 'best_confusion_matrix.npy',confusion_matrix.cpu().numpy())
        else:
            if target_metric_value < best_metric:
                is_best = True
                best_metric = target_metric_value
                save_checkpoint({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'metrics': current_epoch_metrics}, save_path, args.model + "_best")
                np.save(save_path + 'best_confusion_matrix.npy',confusion_matrix.cpu().numpy())
    else:
        is_best=True,
        best_metric = target_metric_value
    return best_metric, is_best

