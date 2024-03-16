# Titanic Survival Prediction Web Application

This Flask web application allows users to predict the survival of passengers on the Titanic based on various input parameters. It also provides exploratory data analysis (EDA) and visualizations of the Titanic dataset.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Ridmi211/survival_predictor_titanic
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt 
    ```
3. Run the Flask application:
    ```bash
    python app.py 
    ```
4. Open your web browser and navigate to http://localhost:5000 to use the application.

## Usage
- Enter the passenger details (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) into the form and click "Predict" to see the predicted survival outcome.
- Explore the "Statistics" page to view descriptive statistics and visualizations of the dataset.
- Visit the "EDA" page to explore missing values, distribution of features, and survival rates.
- Check the "Plot Analysis" page for detailed visualizations of the dataset.
- View the "Correlation" page to see the correlation matrix of numeric features.

## Dataset
The dataset used in this application is the famous Titanic dataset, which contains information about passengers on the Titanic, including whether they survived or not.
