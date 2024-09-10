import torch
import torchvision
import torch
from torch.nn.functional import cross_entropy
import lightning as L
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}

def get_model(model_name, n_classes, bayesian_fg):
    if model_name == "densenet121" and bayesian_fg:
        return BayesDenseNet121(n_classes=n_classes)
    elif model_name == "densenet121":
        return DenseNet121(n_classes=n_classes)
    else:
        raise ValueError(f"Model {model_name} not found")

def DenseNet121(n_classes=3):
    model = torchvision.models.densenet121(weights="DEFAULT")
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    return model

def BayesDenseNet121(n_classes=3):
    model = torchvision.models.densenet121(weights="DEFAULT")
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    dnn_to_bnn(model, const_bnn_prior_parameters)
    print("Bayesian model")
    return model



