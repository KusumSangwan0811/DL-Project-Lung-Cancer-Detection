# DL Project Lung Cancer Detection

## Objective:
The goal of this project is to develop an AI model capable of classifying three types of lung cancer and diagnosing whether a patient has cancer or not. This information can then be used to provide patients with details about their specific type of cancer and potential treatment options.

## Dataset:
The dataset is available on Kaggle: [Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images). It contains images in JPG or PNG format, which can be directly used for model training. The dataset includes images of three types of lung cancer: Adenocarcinoma, Large cell carcinoma, and Squamous cell carcinoma, along with images of normal lung cells. The data is organized into three folders: train, test, and valid, each containing subfolders for the three cancer types and normal CT-Scan images.

## Project Outline:
1. **Import Libraries and Set Up Paths**: Initialize the environment by importing necessary libraries and setting up paths to the dataset.
2. **Load Dataset**: Load the dataset from the specified directories.
3. **Data Preprocessing**: Perform preprocessing steps such as resizing images, normalization, and splitting data into training, validation, and test sets.
4. **Build a Basic CNN Model**: Construct and train an initial Convolutional Neural Network (CNN) model. Evaluate its performance to identify issues like overfitting.
5. **Data Augmentation**: Apply data augmentation techniques to increase the diversity of the training data and improve model generalization.
6. **Improved Network Architecture**: Develop a more complex network with additional convolutional layers and dropout layers to prevent overfitting.
7. **Transfer Learning**: Utilize pre-trained models like VGG16 and a customized SERES VGG16 model to leverage existing knowledge and improve performance.
8. **Training and Evaluation**: Train and evaluate both the improved network and the transfer learning models.
9. **Save the Trained Model**: Save the best-performing model for future use.

By following these steps, the aim is to create a robust model capable of accurately diagnosing lung cancer and providing valuable insights into its type for better treatment planning.


The Customized SERES VGG16 model performs best with an accuracy of 89%.







