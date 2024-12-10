# Deep_Learning_Challenge

# Alphabet Soup Charity Funding Predictor
## Overview
Alphabet Soup, a nonprofit foundation, aims to improve its funding decisions by predicting the success of applicants' ventures. This project leverages machine learning and neural networks to develop a binary classifier that identifies the likelihood of success for funded applicants.

The dataset provided by Alphabet Soup contains over 34,000 records of organizations funded in the past, including metadata such as application type, affiliation, income level, requested funding amount, and whether the funding was successful.

## Features in the Dataset
The dataset contains the following columns:

EIN: Employer Identification Number (ID column)
NAME: Organization name (ID column)
APPLICATION_TYPE: Type of application submitted
AFFILIATION: Sector affiliation
CLASSIFICATION: Government classification of the organization
USE_CASE: Intended use of the requested funding
ORGANIZATION: Type of organization
STATUS: Current operational status
INCOME_AMT: Income classification of the organization
SPECIAL_CONSIDERATIONS: Whether special considerations apply
ASK_AMT: Funding amount requested
IS_SUCCESSFUL: Binary target variable indicating whether the funding was used successfully

# Project Workflow
## Step 1: Preprocess the Data
Before building the model, the dataset needs to be prepared for analysis. Key preprocessing steps include:

Load Data: Import the dataset into a Pandas DataFrame.
Define Features and Target:
Target: IS_SUCCESSFUL
Features: All other columns (excluding EIN and NAME).
Drop Unnecessary Columns: Remove EIN and NAME as they are identifiers without predictive value.
Analyze and Bin Data:
Determine unique value counts for each column.
For columns with more than 10 unique values, group rare categories into an "Other" bin.
Encode Categorical Variables: Use pd.get_dummies() to one-hot encode categorical features.
Split the Data:
Features: X
Target: y
Split into training and testing sets using train_test_split.
Scale Features: Standardize feature values using StandardScaler().

## Step 2: Compile, Train, and Evaluate the Model
A deep learning model is designed using TensorFlow and Keras to predict the binary target variable.

Define the Neural Network Architecture:
Input layer based on the number of features.
At least one hidden layer with an appropriate activation function (e.g., ReLU).
Output layer with a sigmoid activation function for binary classification.
Compile the Model: Use a loss function (e.g., binary crossentropy) and an optimizer (e.g., Adam).
Train the Model: Fit the model on the training data.
Save Model Checkpoints: Create a callback to save model weights every five epochs.
Evaluate the Model: Use the test data to calculate loss and accuracy.
Save Results: Export the trained model as AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
The goal is to improve model accuracy to exceed 75%. The following optimizations can be applied:

Data Adjustments:
Drop or retain additional columns.
Refine binning of rare categories.
Experiment with different group sizes for unique values.
Neural Network Adjustments:
Increase the number of neurons in hidden layers.
Add additional hidden layers.
Experiment with different activation functions (e.g., Tanh, LeakyReLU).
Modify the number of epochs during training.
Iterative Testing:
Perform at least three optimization attempts.
Document each attempt and its outcome.
Save Optimized Model: Export the final model as AlphabetSoupCharity_Optimization.h5.

## File Structure
AlphabetSoupCharity.ipynb: Initial model preprocessing, compilation, training, and evaluation.
AlphabetSoupCharity_Optimization.ipynb: Optimized model with documented iterations and improvements.
AlphabetSoupCharity.h5: Saved initial model.
AlphabetSoupCharity_Optimization.h5: Saved optimized model.

## Tools and Libraries
Pandas: For data manipulation and preprocessing.
scikit-learn: For data splitting and scaling.
TensorFlow/Keras: For building, training, and optimizing the neural network.

## Deliverables
Preprocessed dataset ready for modeling.
Initial trained model (AlphabetSoupCharity.h5).
Optimized model with accuracy improvements (AlphabetSoupCharity_Optimization.h5).
Documentation of methods and outcomes in Jupyter Notebook files.







