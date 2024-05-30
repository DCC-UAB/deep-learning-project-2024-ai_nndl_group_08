[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15063761&assignment_repo_type=AssignmentRepo)
# Business Classification
The task for this project is fine-grained image classification on the [Con-Text dataset](https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html).
The objective is classifying businesses on street view images, combining textual and visual features.

In the ConTextTransformer we first use three pre-trained models for feature extraction: a ResNet50, an OCR, and the FastText word embedding. Then we project the visual (ResNet) and textual (FastText) features to a common dimensionality (Linear Projections) and we feed them into the Tansformer Encoder. Finally, the MLP head takes the output feature of the CLS token and predicts one of the classes with Softmax.

![ConTextTransformer diagram](./ConTextTransformer.png)

## Dependencies

Ensure you have the following libraries installed:

1. **PyTorch**: For building and training neural networks.
2. **TorchVision**: For image transformations and pre-trained models.
3. **einops**: For flexible and readable tensor operations.
4. **Pillow (PIL)**: For image processing.
5. **NumPy**: For numerical operations.
6. **FastText**: For word vector representations.
7. **OS**: For interacting with the operating system (built-in library).
8. **JSON**: For parsing JSON files (built-in library).
9. **Time**: For tracking time (built-in library).

### Installation

You can install the necessary libraries using `pip`. Here are the commands to install these dependencies:

```sh
pip install torch torchvision einops pillow numpy fasttext
```

## How to use it
### Downloading the dataset
First of all, you must download the ConText Dataset. 
```sh
./downloadScript
```

### Training the model (Optional)
Trainig the model usually takes 20 mins+ so you can skip this part by just using the provided *trained_transformer_model.pth*.
```sh
python3 main_train.py
```

### Test the model
```sh
python3 main_test.py
```

### Play around with the Notebook
You can test the model by simply importing your own photos and see how our model performs on prediciting their class. The notebook automatically imports the best transformer model trained by us, along with the necessary pre-trained models, such as ResnNET-50, FastText and the OCR.

The notebook also provides the possibility to evaluate the model on the test set and see its accuracy.


## ConText Dataset
The dataset provided for this task (that can be downloaded with the *downloadScript*) is structured in the following way:

### Images
./data/images/JPEGImages includes the 24.255 images provided. They have been split in 66% for training, 17% for validation and 17% for testing.

### Image names + Target Class
./annotations/split_0.json represents the split of the images in the 3 categories from above. The validation and test set were split so that each one of them contains the same number of instances of each of the 28 classes.

### OCR Labels
./ocr_labels contains the result of the OCR pre-trained model on every image. In the ocr_labels directory, each JSON file represents the result obtained by the OCR on that specific image. There are plenty of images that do not contain any text within them, therefore you will see empty files here and there.

Using the OCR to label each image is a prolonged task, so it was decided to directly upload the resulting OCR labels obtained.

## Code structure
### ConTextTransformer
The ConTextTransformer is a PyTorch-based neural network module that combines convolutional neural networks (CNNs) with transformer models to handle both image and text data. It uses a pre-trained ResNet-50 for image feature extraction and a transformer encoder for processing combined image and text embeddings.

### ConTextDataset
The ConTextDataset class is a custom PyTorch Dataset for handling image and text data. It reads image paths and corresponding text annotations from a JSON file, loads the images, and converts the text to FastText embeddings.

### train.py
This module provides essential functions to train and evaluate a PyTorch model designed to handle both image and text data. The training function runs the model through one epoch of the training data, while the evaluation function assesses the model's performance on a test dataset. These utilities support GPU acceleration if available, ensuring efficient and scalable training and evaluation.

### main_train.py
This script is used for training the Transformer model and saving the best version of it by evaluating its performance on the validation set. The best model is saved, then retrained on the validation set and saved again in the same folder at the path 'trained_transformer_model.pth'. 

One epoch of training usually takes between 40-60 seconds, therefore this script may take some time. For this reason, we uploaded the trained model that resulted from this script. 

### main_test.py
Imports the trained model resulting from the main_train.py script and evaluates its performance on the test_set. 



## Contributors
Balagiu Darian : 1719581@uab.cat  
Micu-Hontan Valentin : 1718971@uab.cat  
Moraru Horia-Andrei : 1720314@uab.cat  

## Bibliography  
https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html
https://github.com/dkaratzas/DL2023-24/blob/main/Problems%2010%20-%20Attention/P10_Attention_Solutions.ipynb
https://e-aules.uab.cat/2023-24/pluginfile.php/609988/course/section/98692/Session%20-%20ProjectsIntroduction.pdf?time=1714423284114
https://github.com/lluisgomez/ConTextTransformer


Deep Learning & Neural Networks __Artificial Intelligence__, 
UAB, 2023
