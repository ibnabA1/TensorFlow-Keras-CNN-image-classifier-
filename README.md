# TensorFlow-Keras-CNN-image-classifier-
This project is a deep learning based binary image classification system that identifies whether a given image contains a car or a motorbike. The model is built using TensorFlow/Keras Convolutional Neural Networks (CNNs) and trained on labeled vehicle images.
he objective of this project is not only to train a neural network but to demonstrate the complete machine learning workflow — from dataset preprocessing to model training, saving, and real-world prediction on unseen images.

Unlike many academic notebooks, this repository converts experimentation into a reusable application.
The trained model can be loaded anytime and used to classify new images without retraining.

This project demonstrates practical understanding of:

Data preprocessing
Image normalization
CNN architecture design
Model training and evaluation
Model persistence
Inference pipeline

Problem Statement

Humans easily distinguish between a car and a bike, but a machine sees only pixel values.
The challenge is to train a neural network to learn visual patterns such as:

Wheel structure
Body shape
Aspect ratio
Spatial features

The CNN automatically extracts these features during training and learns to differentiate between the two classes.

Model Architecture

The classifier uses a Convolutional Neural Network consisting of:

Convolution layers for feature extraction
ReLU activation for non-linearity
MaxPooling layers for dimensional reduction
Flatten layer to convert feature maps to vectors
Dense layers for classification
Sigmoid output for binary classification

The network learns hierarchical features:
Edges → Shapes → Object structure → Vehicle type

Workflow

Load and preprocess dataset

Resize and normalize images

Train CNN model

Evaluate accuracy

Save trained model

Load model for prediction on new images

This transforms the project from an experiment into a deployable ML component.

How To Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python train.py


After training, the model will be saved in:

/model/car_bike_model.h5

3️⃣ Predict on new image
python predict.py path_to_image.jpg


Example Output:

Prediction: Car
Confidence: 0.94

Project Structure
car-bike-classifier/
│
├── model/
│   └── car_bike_model.h5
│
├── notebook/
│   └── cnn_car_bike_classification.ipynb
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md

Technologies Used

Python
TensorFlow / Keras
NumPy
Matplotlib
Pillow (Image Processing)

Key Learning Outcomes

This project demonstrates understanding of:

Designing CNNs for image classification
Preventing overfitting through proper preprocessing
Saving and loading trained models
Building an inference pipeline
Structuring a machine learning repository professionally

Future Improvements

Add more vehicle classes (truck, bus, bicycle)
Convert to real-time camera detection
Deploy using Flask or FastAPI API
Convert model to TensorFlow Lite for mobile devices

Author

Hamza Aftab
Software Engineer | Flutter & AI Enthusiast
