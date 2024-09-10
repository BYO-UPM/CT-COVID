import wandb
import lightning as L
from models import get_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import Accuracy, CatMetric
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from bayesian_torch.utils.util import predictive_entropy, mutual_information
from custom_metrics import AvU, CorrectUncertaintiesConcatenator, IncorrectUncertaintiesConcatenator


def get_lightning_model(model_nm, n_classes, optimizer, lr, momentum, weight_decay, batch_size, n_mc_samples, bayesian_fg=False):
    if bayesian_fg:
        model = get_model(model_nm, n_classes, bayesian_fg=True)
        lightning_model = LitBayesianModel(model, model_nm, n_classes, optimizer, lr, momentum, weight_decay, batch_size, n_mc_samples)
    else:
        model = get_model(model_nm, n_classes, bayesian_fg=False)
        lightning_model = LitModel(model, n_classes, optimizer, lr, momentum, weight_decay)
    return lightning_model

class LitModel(L.LightningModule):
    def __init__(self, model, n_classes, optimizer, lr, momentum, weight_decay):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, batch_idx):
        x, labels = batch
        y_hat = self.forward(x)
        loss = cross_entropy(y_hat, labels)
        predicted_labels = torch.argmax(y_hat, dim=1)
        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch, batch_idx)
        self.train_acc(predicted_labels, labels)
        self.log("train_loss", loss)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False) 
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, labels, predicted_labels = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.val_acc(predicted_labels, labels)
        self.log(
            "val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    
    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"}
    
class LitBayesianModel(L.LightningModule):
    def __init__(self, model, model_nm, n_classes, optimizer, lr, momentum, weight_decay, batch_size, n_mc_samples=10):
        super().__init__()
        self.model = model
        self.model_nm = model_nm
        # print a summary of the model 
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples
        self.save_hyperparameters(ignore=["model"])
        # Train Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.train_precision = MulticlassPrecision(num_classes=n_classes)
        self.train_recall = MulticlassRecall(num_classes=n_classes)
        self.train_f1 = MulticlassF1Score(num_classes=n_classes)
        # Validation Metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes, normalize="all")
        self.val_f1 = MulticlassF1Score(num_classes=n_classes)
        self.val_recall = MulticlassRecall(num_classes=n_classes)
        self.val_precision = MulticlassPrecision(num_classes=n_classes)
        self.val_avu = AvU(uncertainty_threshold=0.5)
        self.val_correct_pred_uncertainties = CorrectUncertaintiesConcatenator()
        self.val_incorrect_pred_uncertainties = IncorrectUncertaintiesConcatenator()
        self.val_correct_model_uncertainties = CorrectUncertaintiesConcatenator()
        self.val_incorrect_model_uncertainties = IncorrectUncertaintiesConcatenator()
        # Test Metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes, normalize="all")
        self.test_f1 = MulticlassF1Score(num_classes=n_classes)
        self.test_recall = MulticlassRecall(num_classes=n_classes)
        self.test_precision = MulticlassPrecision(num_classes=n_classes)
        self.test_avu = AvU(uncertainty_threshold=0.5)
        self.test_correct_pred_uncertainties = CorrectUncertaintiesConcatenator()
        self.test_incorrect_pred_uncertainties = IncorrectUncertaintiesConcatenator()
        self.test_correct_model_uncertainties = CorrectUncertaintiesConcatenator()
        self.test_incorrect_model_uncertainties = IncorrectUncertaintiesConcatenator()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        y_hat = self.forward(x)
        kl_loss = get_kl_loss(self.model)
        ce_loss = cross_entropy(y_hat, labels)
        loss = ce_loss + kl_loss/self.batch_size
        self.log("train_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Separate function to call train metrics with torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            output_mc = []
            for _ in range(self.n_mc_samples):
                logits = self.model(batch[0])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                output_mc.append(probs)
            output = torch.stack(output_mc)
            mean_probs = output.mean(dim=0)
            pred_uncertainty = predictive_entropy(output.data.cpu().numpy())
        self.log("train_acc",
                  self.train_acc(mean_probs, batch[1]), on_step=True, on_epoch=False, prog_bar=True)

    def bayesian_inferece(self, x):
        output_mc = []
        for _ in range(self.n_mc_samples):
            logits = self.model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output_mc.append(probs)
        output = torch.stack(output_mc)
        mean_probs = output.mean(dim=0)
        predicted_labels = torch.argmax(mean_probs, dim=1)
        pred_uncertainty = torch.tensor(predictive_entropy(output.data.cpu().numpy()))
        model_uncertainty = torch.tensor(mutual_information(output.data.cpu().numpy()))
        return predicted_labels, pred_uncertainty, model_uncertainty

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        y_hat = self.forward(x)
        loss = cross_entropy(y_hat, labels)
        predicted_labels, pred_uncertainty, model_uncertainty = self.bayesian_inferece(x)
        self.val_acc(predicted_labels, labels)
        self.val_correct_pred_uncertainties.update(predicted_labels, labels, pred_uncertainty)
        self.val_incorrect_pred_uncertainties.update(predicted_labels, labels, pred_uncertainty)
        self.val_correct_model_uncertainties.update(predicted_labels, labels, model_uncertainty)
        self.val_incorrect_model_uncertainties.update(predicted_labels, labels, model_uncertainty)
        self.log(
            "val_loss" , loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
    
    def log_uncertainty_hist(self, c_pred_uncertainty, i_pred_uncertainty, c_model_uncertainty, i_model_uncertainty, hist_name):
        table_c_pred_uncertainty = wandb.Table(
            data=[[x] for x in c_pred_uncertainty],
            columns=["correct_pred_uncertainty"]
        )
        table_i_pred_uncertainty = wandb.Table(
            data=[[x] for x in i_pred_uncertainty],
            columns=["incorrect_pred_uncertainty"]
        )
        table_c_model_uncertainty = wandb.Table(
            data=[[x] for x in c_model_uncertainty],
            columns=["correct_model_uncertainty"]
        )
        table_i_model_uncertainty = wandb.Table(
            data=[[x] for x in i_model_uncertainty],
            columns=["incorrect_model_uncertainty"]
        )

        hist_c_pred_uncertainty = wandb.plot.histogram(table_c_pred_uncertainty, "correct_pred_uncertainty", title=f"{hist_name}_c_pred_uncertainty")
        hist_i_pred_uncertainty = wandb.plot.histogram(table_i_pred_uncertainty, "incorrect_pred_uncertainty", title=f"{hist_name}_i_pred_uncertainty")
        hist_c_model_uncertainty = wandb.plot.histogram(table_c_model_uncertainty, "correct_model_uncertainty", title=f"{hist_name}_c_model_uncertainty")
        hist_i_model_uncertainty = wandb.plot.histogram(table_i_model_uncertainty, "incorrect_model_uncertainty", title=f"{hist_name}_i_model_uncertainty")
        self.logger.experiment.log({f"{hist_name}_c_pred_uncertainty": hist_c_pred_uncertainty})
        self.logger.experiment.log({f"{hist_name}_i_pred_uncertainty": hist_i_pred_uncertainty})
        self.logger.experiment.log({f"{hist_name}_c_model_uncertainty": hist_c_model_uncertainty})
        self.logger.experiment.log({f"{hist_name}_i_model_uncertainty": hist_i_model_uncertainty})

    def log_confusion_matrix(self, confusion_matrix, name):
        matrix_values = confusion_matrix.compute().numpy()
        # vector of 0 , 1, .. n_classes
        classes = np.arange(matrix_values.shape[0])
        self.logger.experiment.log(
            {name:
            wandb.plots.HeatMap(classes, classes, matrix_values, show_text=True)})

    
    # def on_validation_epoch_end(self):
        # val_correct_pred_uncertainties = [[x] for x in self.val_correct_pred_uncertainties.compute()]
        # table_val_correct_pred_uncertainties = wandb.Table(
        #     data=val_correct_pred_uncertainties,
        #     columns=["val_correct_pred_uncertainties"]
        # )
        # val_incorrect_pred_uncertainties = [[x] for x in self.val_incorrect_pred_uncertainties.compute()]
        # table_val_incorrect_pred_uncertainties = wandb.Table(
        #     data=val_incorrect_pred_uncertainties,
        #     columns=["val_incorrect_pred_uncertainties"]
        # )
        # hist_val_correct_pred_uncertainties = wandb.plot.histogram(table_val_correct_pred_uncertainties, "val_correct_pred_uncertainties", title=f"Epoch {current_epoch}")
        # hist_val_incorrect_pred_uncertainties = wandb.plot.histogram(table_val_incorrect_pred_uncertainties, "val_incorrect_pred_uncertainties", title=f"Epoch {current_epoch}")
        # self.logger.experiment.log({f"val_correct_pred_uncertainties": hist_val_correct_pred_uncertainties})
        # self.logger.experiment.log({f"val_incorrect_pred_uncertainties": hist_val_incorrect_pred_uncertainties})
        # self.val_correct_pred_uncertainties.reset()
        # self.val_incorrect_pred_uncertainties.reset()
    

    def test_step(self, batch, batch_idx):
        x, labels = batch
        predicted_labels, pred_uncertainty, model_uncertainty = self.bayesian_inferece(x)
        self.test_acc(predicted_labels, labels)
        self.test_precision(predicted_labels, labels)
        self.test_recall(predicted_labels, labels)
        self.test_f1(predicted_labels, labels)
        self.test_confusion_matrix(predicted_labels, labels)
        self.test_correct_pred_uncertainties.update(predicted_labels, labels, pred_uncertainty)
        self.test_incorrect_pred_uncertainties.update(predicted_labels, labels, pred_uncertainty)
        self.test_correct_model_uncertainties.update(predicted_labels, labels, model_uncertainty)
        self.test_incorrect_model_uncertainties.update(predicted_labels, labels, model_uncertainty)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_precision", self.test_precision, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_recall", self.test_recall, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_f1", self.test_f1, prog_bar=True, on_epoch=True, on_step=False)

    def on_test_end(self) -> None:
        self.log_uncertainty_hist(
            self.test_correct_pred_uncertainties.compute(),
            self.test_incorrect_pred_uncertainties.compute(),
            self.test_correct_model_uncertainties.compute(),
            self.test_incorrect_model_uncertainties.compute(),
            "test")
        self.log_confusion_matrix(self.test_confusion_matrix, "test_confusion_matrix")
        self.test_correct_pred_uncertainties.reset()
        self.test_incorrect_pred_uncertainties.reset()
        self.test_correct_model_uncertainties.reset()
        self.test_incorrect_model_uncertainties.reset()

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"}

    