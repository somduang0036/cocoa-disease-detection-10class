# Cocoa Image Screening and Leaf–Pod Disease Detection

This repository provides the notebook and training scripts for a two-stage deep learning framework for cocoa image screening and leaf–pod disease detection. **Model 1** uses EfficientNet-B0 to classify input images as **Cocoa** or **Other Plants**. Images classified as Cocoa proceed to **Model 2**, which uses YOLO26n to detect cocoa leaves and pods across 10 classes covering diseases, pest damage, and healthy conditions.

## Dataset

The image dataset contains **22,338 images**, comprising 8,935 Cocoa images and 13,403 Other Plants images. Model 1 uses both groups. Model 2 uses the Cocoa images and their bounding-box annotations.

| File | Contents |
|---|---|
| [images.rar](https://huggingface.co/datasets/tassanee/cocoa-disease-detection-10class/resolve/main/images.rar?download=true) | Image archive containing the training, validation, and test splits |
| [labels.rar](https://huggingface.co/datasets/tassanee/cocoa-disease-detection-10class/resolve/main/labels.rar?download=true) | YOLO bounding-box labels for the detection dataset |

For the complete file listing, see the [Hugging Face Dataset repository](https://huggingface.co/datasets/tassanee/cocoa-disease-detection-10class/tree/main).

## Dataset Setup

1. Download `images.rar` and extract it so that the image folder is `dataset/images/`. Check that the resulting paths match the `image_path` entries in the dataset metadata.
2. To use the standalone YOLO labels, download `labels.rar` and extract it so that the label folder is `dataset/labels/`.
3. Keep `dataset/model2_train_val_labels.zip` in ZIP format. The Model 2 benchmark opens this archive directly and checks the `identity.json` file inside it against the dataset manifest.
4. Preserve the directory name `dataset/matadata_model 2/` exactly as shown. The scripts use this path to locate the class mapping and COCO annotations.

```text
cocoa-diseases/
├── notebook/
│   └── cocoa_deisease_detection.ipynb
├── script/
│   ├── model1_training/
│   │   └── requirements.txt
│   └── model2_training/
│       └── requirements.txt
└── dataset/
    ├── images/                         # Extracted from images.rar
    ├── labels/                         # Optional for standalone YOLO use
    ├── model1_metadata.csv
    ├── model1_metadata.completed.json
    ├── model1_class_mapping.json
    ├── MODEL1_DATASET_VERSION.json
    ├── model2_metadata_detection.csv
    ├── model2_train_val_labels.zip
    └── matadata_model 2/
        ├── class_mapping.json
        ├── annotations_train.coco.json
        ├── annotations_val.coco.json
        └── annotations_test.coco.json
```

## Run the Notebook

Open `notebook/cocoa_deisease_detection.ipynb` in Google Colab and select a GPU runtime. Set the project paths in the setup cells to the directory containing `script/` and `dataset/`. Install the dependencies specified in each model’s `requirements.txt` and create the Model 1 image cache before starting the Model 2 benchmark.

The Model 2 benchmark trains and evaluates YOLOv8m, YOLO11n, YOLO12n, and YOLO26n on the predefined training and validation splits. It saves the overall and per-class validation metrics, comparison plots, individual confusion matrices, and a combined 2×2 confusion matrix. The held-out test set is reserved for final evaluation after validation-based model selection. Scripts create their output directories under `results/` during execution.

The optional Group-Aware Five-Fold Cross-Validation step requires the original YOLO26n baseline `protocol.json` and the prepared `data.yaml` whose SHA-256 hash matches that protocol. This step uses the original training and validation images and excludes the held-out test split.
