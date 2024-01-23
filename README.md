# Streamlit Ritcher's Nepal Earthquake Building Damage Predictor

## Overview
The Streamlit Ritcher's Nepal Earthquake Building Damage Predictor is a web-based application designed to predict the level of damage (low, medium, or high) buildings may sustain in an earthquake scenario, specifically focusing on the Nepal earthquake. This project integrates several machine learning models, including Light Gradient Boosting Machine (LGBM), XGBoost, Neural Network, and TensorFlow Decision Forests, deployed using Streamlit. It provides an interactive interface for users to input building attributes and receive real-time predictions.

## Link to Website
 [Here's a link to the home page the website.](https://michael-ye-ritchers-earthquake-predictor-home.streamlit.app/) Go to the navigation bar to navigate to the other HTML pages for all the different Machine Learning Models to try out the deployed models yourself!

## Technology Stack
- **Programming Language**: Python
- **Libraries**: Streamlit, TensorFlow, LightGBM, XGBoost
- **Web Technologies**: HTML, CSS (used within Streamlit framework)
- **Deployment**: Streamlit Sharing (for cloud deployment)

## Models Description
Each model offers unique insights and predictions based on the user's input:

### Light Gradient Boosting Machine (LGBM)
- Fast, efficient, and highly accurate.
- Ideal for handling large datasets with a focus on speed and performance.

### XGBoost
- An optimized distributed gradient boosting library.
- Known for its efficiency, flexibility, and portability.

### Neural Network
- Implements a deep learning approach.
- Capable of capturing complex patterns in data.

### TensorFlow Decision Forests
- A TensorFlow-based implementation of Random Forests and other decision tree-based models.
- Provides an intuitive understanding of the prediction process.

## Web Deployment
The models are deployed on a Streamlit-based web interface. Users can interact with the models by inputting building attributes (categorical, numerical, and binary data types). The interface is intuitive and user-friendly, ensuring ease of use for non-technical users.

## User Interface
The main webpage provides a navigation menu to access each model's page. Users can input various building attributes, such as age, area, height, and material type. After submission, the prediction is displayed on the website.

## Data Preprocessing
Each model has its data preprocessing script (`data_preprocessing_LGBM.py`, etc.) to format user inputs correctly. These scripts ensure that the input data is compatible with the respective models' requirements.

## Repository Structure
- Python Scripts: Model and preprocessing scripts (`LGBM.py`, `data_preprocessing_LGBM.py`, etc.)
- HTML Files: Individual pages for each model (`LGBM_1.html`, etc.)
- Main Files: `index.html`, `index.py` for the homepage and app setup.
- Saved Models: Pre-trained models ready for predictions (`saved_LGBM_model7.joblib`, etc.)

## Installation and Usage
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run `streamlit run index.py` to start the application.
4. Access the web interface through the provided local URL.

## Conclusion
This project was an invaluable learning experience in model deployment, integrating HTML with Streamlit, and handling diverse data types. It showcases the practical application of machine learning models in predicting real-world scenarios.


Certainly! I'll include a link to the homepage of your Streamlit application in the README.md. Here's the updated section with the link:

