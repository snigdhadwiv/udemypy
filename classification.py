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

st.write("Data Analysis")

fig, ax= plt.subplots()
sns.histplot(x='sepal_length', y='petal_width',
data=df)
st.pyplot(fig)



from sklearn.model_selection import train_test_split

X=df[['sepal_length','sepal_width','petal_length','petal_width']]
y=df['species']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, LabelEncoder

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
total_accuracy_score=accuracy
st.write("model is now encoded, scaled, imputed, and random forest is put on the cleaned preprocessed dataset. total accuracy score is: ",total_accuracy_score)
