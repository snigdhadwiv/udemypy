import pandas as pd
import numpy as np 
import streamlit as st

st.title("Sniggy ki golfball practice")
list=[1,2,3,4,5,6,7,8,9]
st.write("this is a simple golfball practice, take it easy sniggy")
text_input=st.text_input("enter your name if you wanna be friends with sniggy")
if text_input=="somu":
    st.write("you're her best friend")
elif text_input=="mumma":
    st.write("you're her kuchhu puchhu best momma")
elif text_input=="papa":
    st.write("your her life and her world best papa")

else: st.write("okay you're now her new friend")
age=st.slider("how old r u monke: ", 0,100,50)
st.write(f"you are {age} years old which makes you qualified to be sniggy's besto FRIENDO")
st.image('besto_friendo.jfif')


