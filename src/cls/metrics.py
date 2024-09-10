import torch
import wandb

class Metrics():
    def __init__(self, use_wandb=False, n_classes=3):
        self.data = list()
        self.use_wandb = use_wandb
        self.n_classes = n_classes

    def empty_metric_dict(self):
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0,
            "loss": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "confusion_matrix": torch.zeros((self.n_classes, self.n_classes))
        }

    def get_epoch_metrics(self, epoch, mode):
        metrics = [x["metrics"] for x in self.data if x["epoch"] == epoch and x["mode"] == mode]
        return metrics[0]
    
    def initialize_empty_epoch_metrics(self, epoch, mode):
        metrics = self.empty_metric_dict()
        self.data.append(
            {
                "epoch": epoch,
                "mode": mode,
                "metrics": metrics
            }
        )
        
    # Reads all metrics and sums the values for each epoch and mode
    def update_minibatch_metrics(self, epoch, mode, **kwargs):
        current_epoch_metrics = self.get_epoch_metrics(epoch, mode)
        for key, value in kwargs.items():
            current_epoch_metrics[key] += value
        self.data.append(
            {
                "epoch": epoch,
                "mode": mode,
                "metrics": current_epoch_metrics
            }
        )

    def log_epoch_metrics(self, epoch, mode, total_batches, print=True):
        current_epoch_metrics = self.get_epoch_metrics(epoch, mode)
        current_epoch_metrics["accuracy"] = current_epoch_metrics["correct"] / current_epoch_metrics["total"]
        current_epoch_metrics["precision"] = current_epoch_metrics["correct"] / (current_epoch_metrics["correct"] + (current_epoch_metrics["total"] - current_epoch_metrics["correct"]))
        current_epoch_metrics["recall"] = current_epoch_metrics["correct"] / (current_epoch_metrics["correct"] + (current_epoch_metrics["total"] - current_epoch_metrics["correct"]))
        current_epoch_metrics["f1"] = 2 * (current_epoch_metrics["precision"] * current_epoch_metrics["recall"]) / (current_epoch_metrics["precision"] + current_epoch_metrics["recall"])
        current_epoch_metrics["loss"] = current_epoch_metrics["loss"] / total_batches
        self.data.append(
            {
                "epoch": epoch,
                "mode": mode,
                "metrics": current_epoch_metrics
            }
        )
        if print:
            self.print_metrics(epoch, mode)
        if self.use_wandb:
            self.log_wandb_epoch_metrics(epoch, mode)
    
    def log_final_metrics(self, epoch, mode):
        if self.use_wandb:
            self.log_wandb_confusion_matrix(epoch, mode)
        print(f"Final metrics - {mode}")
        self.print_metrics(epoch, mode)

    def print_metrics(self, epoch, mode,):
        metrics = self.get_epoch_metrics(epoch, mode)
        print(f"Epoch {epoch} - {mode} - Loss {metrics['loss']:.4f} - Accuracy {metrics['accuracy']:.4f} - Precision {metrics['precision']:.4f} - Recall {metrics['recall']:.4f} - F1 {metrics['f1']:.4f}")
    
    # Wandb related functionalities
    def log_wandb_epoch_metrics(self, epoch, mode):
        metrics = self.get_epoch_metrics(epoch, mode)
        wandb.log({
            f"{mode}_loss": metrics["loss"],
            f"{mode}_accuracy": metrics["accuracy"],
            f"{mode}_precision": metrics["precision"],
            f"{mode}_recall": metrics["recall"],
            f"{mode}_f1": metrics["f1"],
        })

    def log_wandb_confusion_matrix(self, epoch, mode):
        metrics = self.get_epoch_metrics(epoch, mode)
        preds, labels = self.confusion_matrix_to_preds(metrics["confusion_matrix"])
        wandb.log({
            f"{mode}_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                    y_true=labels,
                                                                    preds=preds,
                                                                    class_names=["normal", "common-pneumonia", "covid"])
        })
    
    
    def confusion_matrix_to_preds(self, confusion_matrix, to_tensor=False):
        preds = list()
        labels = list()
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                matrix_value = confusion_matrix[i][j]
                for _ in range(int(matrix_value)):
                    preds.append(j)
                    labels.append(i)
        if to_tensor:
            preds = torch.tensor(preds)
            labels = torch.tensor(labels)
        return preds, labels


    @staticmethod
    def accuracy(output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct, len(target), correct / len(target)
    
    def confusion_matrix(self, output, labels):
        conf_matrix = torch.zeros((self.n_classes, self.n_classes))
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            for t, p in zip(labels.view(-1), pred.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
        return conf_matrix