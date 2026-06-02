```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# TITLE & DATA LOADING
# =========================

st.title("Iris Classification using Random Forest")
st.write("Let's perform Classification in Iris Dataset of Seaborn")

@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

# =========================
# DATA PREVIEW
# =========================

st.subheader("Dataset Sample")
st.write(df.sample(5))

st.write("Dataset Shape:", df.shape)

# =========================
# BEFORE OUTLIER HANDLING
# =========================

st.subheader("Boxplot Before Outlier Handling")

fig, ax = plt.subplots()
sns.boxplot(
    data=df.select_dtypes(include="number"),
    ax=ax
)
st.pyplot(fig)

st.write("Visible outliers can be seen in sepal_width.")

# =========================
# OUTLIER HANDLING
# =========================

Q1 = df["sepal_width"].quantile(0.25)
Q3 = df["sepal_width"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df["sepal_width"] = df["sepal_width"].clip(
    lower=lower,
    upper=upper
)

# =========================
# AFTER OUTLIER HANDLING
# =========================

st.subheader("Boxplot After Outlier Handling")

fig, ax = plt.subplots()
sns.boxplot(
    data=df.select_dtypes(include="number"),
    ax=ax
)
st.pyplot(fig)

st.write("Outliers handled using IQR clipping.")

# =========================
# DATA ANALYSIS
# =========================

st.subheader("Data Analysis")

fig, ax = plt.subplots()

sns.histplot(
    data=df,
    x="sepal_length",
    y="petal_width",
    ax=ax
)

st.pyplot(fig)

st.write("Missing Values")
st.write(df.isnull().sum())

st.write("Data Types")
st.write(df.dtypes)

st.write("Columns")
st.write(df.columns)

# =========================
# TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# ENCODING
# =========================

from sklearn.preprocessing import StandardScaler, LabelEncoder

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

# =========================
# ACCURACY
# =========================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.success(
    f"Model Accuracy: {accuracy:.2%}"
)

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Predict a Flower")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    float(df["sepal_length"].min()),
    float(df["sepal_length"].max()),
    float(df["sepal_length"].mean())
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    float(df["sepal_width"].min()),
    float(df["sepal_width"].max()),
    float(df["sepal_width"].mean())
)

petal_length = st.sidebar.slider(
    "Petal Length",
    float(df["petal_length"].min()),
    float(df["petal_length"].max()),
    float(df["petal_length"].mean())
)

petal_width = st.sidebar.slider(
    "Petal Width",
    float(df["petal_width"].min()),
    float(df["petal_width"].max()),
    float(df["petal_width"].mean())
)

# =========================
# USER INPUT DATAFRAME
# =========================

input_data = pd.DataFrame(
    {
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width],
    }
)

st.subheader("Current Input Values")
st.write(input_data)

# =========================
# PREDICTION
# =========================

if st.button("Predict Species"):

    input_scaled = scaler.transform(input_data)

    prediction = rfc.predict(input_scaled)

    predicted_species = encoder.inverse_transform(
        prediction
    )

    st.success(
        f"Predicted Species: {predicted_species[0]}"
    )
