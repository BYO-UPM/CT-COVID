from pytorch_lightning import LightningModule
from models_lightning import LitBayesianModel, LitModel
from models import get_model

def main():
    print("tests")
    model_checkpoint = "/media/my_ftp/TFTs/amoure/results/cls/test/lightning_logs/version_0/checkpoints/densenet121best-epoch=05-val_loss=1.00.ckpt"
    model_nm = "densenet121"
    _model_by = get_model(model_nm, 3, bayesian_fg=True)

    loaded_bayesian_model = LitBayesianModel.load_from_checkpoint(model_checkpoint, model = _model_by,n_classes=3, optimizer="sgd", lr=1e-4, momentum=0.9, weight_decay=1e-4, bayesian_fg=True)
    # Print model summary
    print(loaded_bayesian_model)
    # Print all model parameters from the first dense layer
    print(loaded_bayesian_model.model.features.denseblock1.denselayer1.state_dict())
    
if __name__ == "__main__":
    main()