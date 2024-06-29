# DL-Project-Lung-Cancer-Detection
The objective is to classify (3 types) and diagnose, if the patient have cancer or not with the help of AI model. So that the information about the type of cancer and the way of treatment can be provided to them.


##About the Dataset:
This dataset is available on Kaggle on the link https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

Images are in jpg or png format and thus can be used directly to fit the model. This dataset contain 3 chest cancer types: Adenocarcinoma,Large cell carcinoma, Squamous cell carcinoma, and 1 folder for the normal cell. Data folder consists of train, test, and valid folders. Each folder contain 3 folders of different chest cancer types (adenocarcinoma,large cell carcinoma,squamous cell carcinoma) and 1 folder of normal CT-Scan images (normal).


##Outline of Project:
1. Import Libraries, Set up path and Load dataset.
2. Data Preprocessing.
3. Build a basic CNN model, Train and evaluate it. Model was overfitting on train data.
4. Data Augmentation to increase the diversity of the training data.
5. Use Improved Network Architecture which is a deeper network with more convolutional layers and additional dropout layers to prevent overfitting.
6. Transfer Learning Using VGG16 model and Customized SERES VGG16 Model.
7. Train and evaluate for both models.
8. Save the trained model.









