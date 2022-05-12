import streamlit as st
from dalle import display_images

st.title("Dall-E")

text = st.text_input("Text to make an image from:")

num_images = st.slider("Number of images to generate:", 1, 10, 1)

ok = st.button("Make images")

if ok:
    display_images(text, num_images)