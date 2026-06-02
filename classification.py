import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

#title portion with importing the dataset
st.title("Classification")
st.write("Let's perform Classification in Iris Dataset of Seaborn")
@st.cache_data
def load_data():
    return sns.load_dataset('iris')
df=load_data()


 #to not go through the hassle of loading data everytime 
#shape and boxplot of full df
st.write(df.sample(5))
st.write(df.shape)

fig, ax = plt.subplots()
sns.boxplot(data=df.select_dtypes(include='number'), ax=ax)
st.pyplot(fig)


st.write("we have some visible outliers in sepal_width")

#handling outliers 
st.write("datatype of sepal width is:", df["sepal_width"].dtype)

mean=df['sepal_width'].mean()
Q1=df['sepal_width'].quantile(0.25)
Q3=df['sepal_width'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
df['sepal_width']=df['sepal_width'].clip(lower=lower, upper=upper)


fig, ax = plt.subplots()
sns.boxplot(data=df.select_dtypes(include='number'), ax=ax)
st.pyplot(fig)
st.write("hopefully the outliers are gone by now")

df["sepal_width"] = df["sepal_width"].clip(lower=lower, upper=upper)

