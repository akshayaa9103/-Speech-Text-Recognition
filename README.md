# Speech-Text-Recognition
 This repository contains the implementation of a Speech-to-Text digit recognition system using deep learning techniques. The system is trained on an open-source dataset of 30,000 English-spoken digits from 60 different speakers. The model employs Mel-Frequency Cepstral Coefficients (MFCC) for feature extraction, followed by a three-layer Deep Dense Neural Network (DNN) with ReLU activation for classification. The system also integrates Explainable Artificial Intelligence (XAI) through Local Interpretable Model-agnostic Explanations (LIME) to improve transparency and interpretability.
## Table of Contents
* Introduction
* Project Structure
* Dataset
* Features
* Methodology
* Model Training
* Results
* Evaluation Metrics
* Contributing
* License
## Introduction
This project aims to implement a robust Speech-to-Text system capable of accurately recognizing spoken digits. The model leverages deep learning methodologies, particularly a three-layer deep dense neural network with ReLU activation for classification. The feature extraction process involves MFCC to capture essential characteristics of audio signals, and LIME is used to enhance model transparency by offering explanations for its predictions.
## Project Structure
├── dataset/               # Directory containing the audio dataset and metadata
├── src/                   # Source code for training and testing the model
│   ├── preprocess.py      # Data preprocessing and feature extraction (MFCC)
│   ├── model.py           # Model architecture and training code
│   ├── evaluate.py        # Code to evaluate model performance
│   ├── lime_explainer.py  # LIME implementation for model interpretation
│   └── utils.py           # Utility functions for training and data handling
├── output/                # Trained models and evaluation results
│   ├── model_weights.h5   # Trained model weights
│   ├── confusion_matrix.png  # Confusion matrix image
│   └── results.json       # Evaluation results in JSON format
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # Project license
## Dataset
The dataset used for training is AudioMNIST, which contains 30,000 audio samples recorded by 60 different speakers, representing spoken digits from 0 to 9. The dataset is publicly available on Kaggle (https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist). The audio samples are accompanied by metadata, including speaker gender and age.
## Features
* MFCC Feature Extraction: Extracts essential features from audio signals to enable the model to learn from them effectively.
* Deep Neural Network (DNN): A three-layer DNN with ReLU activation is used for classification, with a softmax output layer for multi-class classification.
* Explainable AI (LIME): Local Interpretable Model-agnostic Explanations (LIME) is applied to interpret the model's decisions and provide transparency.
* Evaluation Metrics: The model is evaluated using precision, recall, F1 score, accuracy, and ROC-AUC to assess its performance.
## Methodology
1. Data Preprocessing: The audio signals are preprocessed by resampling, normalizing, and removing noise. MFCC is then applied to extract key features.
2. Model Architecture: A deep neural network with three hidden layers and ReLU activation is used to classify the audio samples. The softmax output layer enables multi-class classification.
3. Optimization: The Adam optimizer is used to minimize the cross-entropy loss and optimize the model's performance.
4. Explainability: LIME is integrated to explain the model's predictions by approximating the model locally with an interpretable linear model.
## Model Training
1. Data Split: The dataset is split into training (70%) and testing (30%) sets.
2. Training Process: The deep neural network is trained on the preprocessed data using the Adam optimizer to minimize the loss.
3. Model Evaluation: After training, the model is evaluated on the test set using various metrics like accuracy, precision, recall, F1 score, and AUC-ROC.
## Results
* The model achieved an impressive accuracy of 99.14% on the test set, showcasing its high performance in digit recognition.
* The confusion matrix and ROC curve demonstrate the model's ability to discriminate between different digits.
* The use of LIME allows for transparent model predictions, showing how each feature contributes to the final decision.
## Evaluation Metrics
The following metrics were used to evaluate the model:
* Precision: The ratio of correctly predicted positive observations to the total predicted positives.
* Recall: The ratio of correctly predicted positive observations to the total actual positives.
* F1 Score: The weighted average of precision and recall.
* Accuracy: The proportion of correct predictions.
* ROC-AUC: The area under the ROC curve, assessing the model's discriminatory ability.
## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository, create a new branch, and submit a pull request with your changes.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact If you have any questions or need further assistance, please contact me via akshayaas910@gmail.com .
