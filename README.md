Here’s a well-structured `README.md` tailored for your repo, summarizing your paper and making it easy for new users to understand, cite, and run the main training functionalities. Please adapt sections as your repo evolves.

---

# Bayesian Screening and Explainable Localization of Lung Lesions from CT Scans

**Author:** Álvaro Moure Prado  
**Correspondence:** ignacio.godino@upm.es  
**License:** CC-BY-NC 4.0 International  
**bioRxiv Preprint:** [doi:10.1101/2025.04.08.647710](https://doi.org/10.1101/2025.04.08.647710)

## Overview

This repository accompanies the paper:

> **Bayesian Screening and Explainable Localization of Lung Lesions from CT Scans**  
> *Álvaro Moure Prado et al., bioRxiv, April 2025*

### Abstract

While semantic segmentation allows precise lesion localization, bounding box-based object detection is considered more effective for highlighting target regions without replacing clinical expertise—reducing attentional and automation biases. This work proposes a two-stage approach for more explainable detection of pneumonia lesions in lung CT scans:
- **Stage 1:** Bayesian uncertainty-driven screening classifies each CT slice for disease presence.
- **Stage 2:** Lesion localization is applied to screened positive images using object detection architectures.

Key contributions:
- **Explainability:** Provides confidence measures for both predictions and regions of interest.
- **Expert-centric:** Supports, but does not replace, clinical judgement.
- **Methodological innovation:** Introduces a fusion strategy to merge overlapping bounding boxes, improving localization of scattered lesions.

Experiments were conducted on ~90,000 CT images from public COVID-19, bacterial, fungal, viral pneumonia datasets and controls.  
- **Screening Accuracy for COVID-19:** up to 94.61%  
- **Lesion Localization mAP50:** 0.5559  
- **Models:** Bayesian DenseNet121, YOLOv8, Cascade R-CNN, RetinaNet

Results generalize to other pneumonia types.

---

## Table of Contents

- [Background](#background)
- [Methods](#methods)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start: Training & Inference](#quick-start-training--inference)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Background

CT imaging provides high sensitivity for pneumonia diagnosis, especially COVID-19, enabling accurate localization of lung lesions. However, manual analysis is time-consuming and subjective. AI-powered systems can assist radiologists by automating screening and lesion localization, provided they communicate model uncertainty and do not induce clinical over-reliance.

This repo implements and benchmarks:
- **Bayesian screening** (DenseNet121 with uncertainty quantification)
- **Object detection-based localization** (YOLOv8, Cascade R-CNN, RetinaNet)
- **Bounding box fusion** to handle overlapping and scattered lesion regions

---

## Methods

- **Screening:**  
  Bayesian DenseNet121 classifies CT slices, estimating predictive uncertainty via Monte Carlo dropout (BayesianTorch).
- **Localization:**  
  Positive slices are processed by YOLOv8, Cascade R-CNN, or RetinaNet to detect lesions as bounding boxes.
- **BB Fusion:**  
  Semantic segmentation masks are converted to bounding boxes using connected components, Non-Maximum Suppression, and HDBSCAN clustering.

**Pipeline:**  
1. Pre-process images (CLAHE, histogram equalization)
2. Screen slices for disease (COVID-19, CAP, HC)
3. Apply object detector to positive cases
4. Post-process bounding boxes for robust lesion localization

---

## Datasets

Eight public datasets were curated and harmonized with segmentation mask-to-bounding box conversion.  
- **COVID-19, CAP, HC, and control subjects**
- Details and code for annotation conversion provided in the repo.

See [the paper](https://doi.org/10.1101/2025.04.08.647710) for dataset specifics.

---

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/alvaromoureupm/<repo-name>.git
   cd <repo-name>
   ```

2. **Install dependencies:**  
   (Recommended: use a Python 3.9+ virtual environment)
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets:**  
   - Depending on the time of reading, some data sources might have changes its access policies.

---

## Quick Start: Training & Inference

### 1. Dataset Creation

For dataset creation please check:
```data_processing/gen_classification_dataset.ipynb``` 

and

```data_processing/gen_object_detection_dataset.ipynb``` 

Original CT Images need to be downloaded from the original sources independently.


### 2. Screening (Bayesian DenseNet121)

Train screening module:
```bash
python main_lightning.py **kwargs
```
- Configurable for deterministic or Bayesian mode.
- Outputs accuracy, ROC curves, and uncertainty plots.

### 3. Lesion Localization (YOLOv8 / Cascade R-CNN / RetinaNet)

Train lesion localization module (MMDET-based):
```bash
python src/det/models/mmdet/train.py
```

Train lesion localization module (YOLO-based):
```bash
src/det/models/yolo/train_yolo.py
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{moureprado2025bayesian,
  author    = {Álvaro Moure Prado, Alejandro Guerrero-López, Julián D. Arias-Londoño, and Juan I. Godino-Llorente},
  title     = {Bayesian Screening and Explainable Localization of Lung Lesions from CT Scans},
  journal   = {bioRxiv},
  year      = {2025},
  doi       = {10.1101/2025.04.08.647710}
}
```

---

## License

This repository is licensed under [CC-BY-NC 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

---

## Contact

For questions and collaborations, please contact:  
**ignacio.godino@upm.es**

---

**Keywords:** Pneumonia | COVID-19 | Lung Lesion Localization | CT Scan | Explainable AI | Bayesian Deep Learning | Object Detection | Decision Support System

---

*For further details, implementation, and visualizations, read the full preprint and explore the code and notebooks provided in this repository.*

---

Let me know if you want specific code snippets, a more detailed usage section, or additional badges!
