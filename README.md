# https://irissimpleclassification.streamlit.app/
basic classification on iris dataset with 1.0 accuracy, code fully written on streamlit.
## Iris Classification Web App

A machine learning web application built with Streamlit that performs classification on the Iris dataset using a Random Forest Classifier. The application includes exploratory data analysis, outlier handling, preprocessing, model training, evaluation, and real-time species prediction through an interactive user interface.

## Features

* Iris dataset loading using Seaborn
* Data exploration and visualization
* Outlier detection and handling using the IQR method
* Data preprocessing with:

  * Label Encoding
  * Standard Scaling
* Random Forest Classification
* Model accuracy evaluation
* Interactive prediction system using Streamlit sliders
* Real-time flower species prediction

## Dataset

The project uses the Iris dataset, which contains measurements of iris flowers:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Target Classes:

* Setosa
* Versicolor
* Virginica

## Machine Learning Workflow

1. Load Dataset
2. Data Exploration
3. Outlier Detection
4. Outlier Treatment using IQR Clipping
5. Train-Test Split
6. Label Encoding
7. Feature Scaling
8. Random Forest Training
9. Model Evaluation
10. User Prediction Interface

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-Learn

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/udemypy.git
cd udemypy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run classification.py
```

## Requirements

```text
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Application Preview

The application provides:

* Dataset preview
* Boxplots before and after outlier handling
* Data analysis visualizations
* Model accuracy score
* Interactive sliders for feature input
* Live species prediction

## Project Structure

```text
udemypy/
│
├── classification.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Deployment

The application is deployed using Streamlit Community Cloud.

To deploy:

1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `classification.py` as the main file.
5. Deploy the application.

## Future Improvements

* Multiple classifier comparison
* Hyperparameter tuning
* Confusion matrix visualization
* Feature importance analysis
* Downloadable prediction reports
* Additional datasets and classification models

## Author

Snighda

Student | Data Science & Machine Learning Enthusiast
