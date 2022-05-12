import requests
import base64
import streamlit as st


URL = "https://six-tires-cheat-35-197-74-75.loca.lt"
headers = {
    'Bypass-Tunnel-Reminder': "go",
    'mode': 'no-cors'
}

def check_if_valid_backend(url):
    try:
        resp = requests.get(url, timeout=5, headers=headers)
        return resp.status_code == 200
    except requests.exceptions.Timeout:
        return False

def call_dalle(url, text, num_images=1):
    data = {"text": text, "num_images": num_images}
    resp = requests.post(url + "/dalle", json=data, headers=headers)
    if resp.status_code == 200:
        return resp
    
def display_images(text, num_images=1):
    valid = check_if_valid_backend(URL)
    if not valid:
        st.write("Backend is not available")
    resp = call_dalle(URL, text, num_images)
    if resp is not None:
        for data in resp.json():
            img_data = base64.b64decode(data)
            st.image(img_data)