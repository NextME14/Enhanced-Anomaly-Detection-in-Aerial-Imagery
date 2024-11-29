# Enhanced Anomaly Detection in Aerial Imagery using Synthetic Data Generation
## Overview
This project aims to enhance anomaly detection in aerial imagery through the use of synthetic data generation. The unCLIP model is utilized to generate high-quality, context-specific images from textual descriptions, which are then used to train the YOLOv8 model for anomaly detection in aerial images. This approach addresses the challenge of limited real-world data by providing a flexible and robust dataset that can be used for training machine learning models to detect a variety of anomalies such as illegal logging, infrastructure damage, and environmental disasters.

## Table of Contents
1. Project Description
2. Installation Instructions
3. Usage
4. Methodology
5. Results
6. Discussion


### 1. Project Description
This repository contains the code and methods used to generate synthetic aerial images using the unCLIP model and train the YOLOv8 model for anomaly detection. The workflow includes:

- Synthetic Image Generation using unCLIP: Over 70 textual prompts are crafted to describe various types of anomalies (e.g., illegal logging, chemical spills). These prompts guide the unCLIP model to generate high-resolution images.
* Image Annotation and Preprocessing: The generated images are annotated using pixel-level segmentation masks and then preprocessed to improve the dataset's variability.
+ Training YOLOv8 for Anomaly Detection: The annotated and preprocessed dataset is used to train the YOLOv8 model, which is then validated and tested for anomaly detection performance.

### 2. Installation Instructions
#### Prerequisites
- Python 3.8 or higher
- pip (need pip to install the required libraries)
* GPU support (optional for faster training)
+ Dependencies listed in `requirements.txt`

##### Create the `requirements.txt` file:
This file will list all the Python libraries and packages required to run the project.

```

ultralytics==8.0.196
roboflow==0.2.1
diffusers["torch"]==0.10.0
transformers==4.11.3
accelerate==0.5.1

```

#### Clone the Repository
```
git clone https://github.com/your-username/enhanced-anomaly-detection.git
cd enhanced-anomaly-detection
```

#### Install Dependencies
Install the required Python libraries:
```

pip install -r requirements.txt

```

#### Install YOLOv8

```

pip install ultralytics==8.0.196

```

#### Install Roboflow

To download the dataset from Roboflow:

```

pip install roboflow

```

#### Install Diffusers and Transformers

To use the unCLIP model, install the following:

```

pip install diffusers["torch"] transformers
pip install accelerate

```

### 3. Usage

#### Generate Synthetic Images using unCLIP
The first step is generating synthetic images from textual prompts. This is achieved by running the following command:

```

python generate_synthetic_images.py

```
This script will generate a set of synthetic images based on the predefined prompts and save them in the generated_images/ folder.

#### Annotate and Preprocess Images
Once the synthetic images are generated, you need to annotate them using Roboflow for training YOLOv8. You can upload the images to Roboflow, manually annotate them with segmentation masks, and export the dataset in the YOLOv8 format.

#### Train YOLOv8 for Anomaly Detection
After preparing the dataset, you can train the YOLOv8 model using the following command:

```

python train_yolov8.py

```

This will train the YOLOv8 model on the synthetic data for 25 epochs.

#### Evaluate the Model
To evaluate the model's performance:

```

python evaluate_model.py

```
This will assess the model's ability to detect anomalies and display key metrics like precision, recall, and mAP.

#### Validate the Custom Model
To validate the trained model using new data:

```

yolo task=segment mode=val model={HOME}/runs/segment/train/weights/best.pt data={dataset.location}/data.yaml

```

This step evaluates the model's ability to generalize and detect anomalies in new images.

#### Deploy the Model
To deploy the trained model and integrate it into a production environment:

```

project.version(dataset.version).deploy(model_type="yolov8-seg", model_path=f"{HOME}/runs/segment/train/")

```
### 4. Methodology
This project integrates generative AI (unCLIP) for synthetic data generation and YOLOv8 for anomaly detection. The methodology is broken into the following steps:

- Generate Synthetic Aerial Images: Over 70 text prompts are used to generate synthetic images representing various anomalies.
- Image Annotation: Images are annotated with pixel-level segmentation masks to identify anomalies.
- Preprocessing: Resizing, normalization, and data augmentation techniques are applied to improve the dataset.
- Training YOLOv8: The annotated dataset is used to train YOLOv8 for anomaly detection.
- Evaluation and Deployment: The model's performance is evaluated, and the trained model is deployed for real-time anomaly detection.

### 5. Results
The results from the training process indicate a significant improvement in anomaly detection, with high precision and recall scores achieved. The use of synthetic data from unCLIP demonstrated the ability to simulate various anomaly scenarios, which contributed to a more robust training process.

#### Key Results:
- SSIM: 0.0932 (indicating room for improvement in texture fidelity)
- PSNR: 27.859 dB (showing that the generated images were realistic but still had noticeable noise)

#### YOLOv8 Performance:
- Significant improvement in precision, recall, and mAP scores over 25 epochs.
- Ability to detect and segment anomalies effectively.

### 6. Discussion
The synthetic data generated by unCLIP proved valuable for training the YOLOv8 model, despite some limitations in image fidelity. Future improvements could focus on enhancing image realism, expanding the dataset, and refining preprocessing techniques to improve model generalization further.

## Acknowledgments
- HuggingFace for hosting the unCLIP model. https://huggingface.co/kakaobrain/karlo-v1-alpha
  
- Roboflow for providing image annotation tools.
- Ultralytics for YOLOv8.
