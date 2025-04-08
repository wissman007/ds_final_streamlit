import streamlit as st
import numpy as np
import cv2
import requests
from os import path, environ
import pandas as pd

def get_byte_img_shape(image: bytes):
    img_full_byte = np.array(bytearray(image), dtype=np.uint8)
    img_full_cv = cv2.imdecode(img_full_byte, 1)
    height, width, _ = img_full_cv.shape
    return height, width

def get_img_chara(image: bytes, file_ext:str, split:str):
    pre = None
    post = None
    img_full_byte = np.array(bytearray(image), dtype=np.uint8)
    img_full_cv = cv2.imdecode(img_full_byte, 1)
    height, width, _ = img_full_cv.shape
    if split == 'h':
        pre = img_full_cv[0:int((height/2)+1), 0:width]
        post = img_full_cv[int(height/2):height, 0:width]
        height = int(height / 2)
    else:
        pre = img_full_cv[0:height, 0:int(width/2)]
        post = img_full_cv[0:height, int((width/2)+1):width]
        width = int(width / 2)
    _, pre_buff = cv2.imencode(file_ext, pre)
    pre_bytes = pre_buff.tobytes()
    _, post_buff = cv2.imencode(file_ext, post)
    post_bytes = post_buff.tobytes()
    files = [('files', pre_bytes), ('files', post_bytes)]
    return files, height, width, pre, post

def resized_response(content: bytes, width: int, height: int):
    mask = content
    mask_nd = np.array(bytearray(mask), dtype=np.uint8)
    mask_cv = cv2.resize(cv2.imdecode(mask_nd, 1), (width, height))
    return mask_cv

pre_bytes = None
post_bytes = None
pc_height = 0
pc_width = 0

labels = {'Severity': ['No damage', 'Minor damage', 'Major damage', 'Destroyed']}
# labels = {'s1': ['No damage'], 's2': ['Minor damage'], 's3': ['Major damage'], 's4': ['Destroyed']}
df = pd.DataFrame(labels)
df = df.style.map(lambda x: f"background-color: {'cyan' if 'No' in x else ('yellow' if 'Min' in x else ('orange' if 'Maj' in x else 'red')) }")

st.title("Damage estimator")

st.text("Load the couple of images of the disaster photo you want to load")

tab1, tab2, tab3 = st.tabs(["Load from file", "Load from URL", "Load from dual image URL"])

# Tab 1 - FROM FILE
col1, col2 = tab1.columns(2)

pre_disaster_file = col1.file_uploader("Choose a pre-disaster image file", type="png")
post_disaster_file = col2.file_uploader("Choose a post-disaster image file", type="png")

if pre_disaster_file is not None:
    # Convert the file to an opencv image.
    pre_bytes = pre_disaster_file.read()
    file_bytes = np.asarray(bytearray(pre_bytes), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    pc_height, pc_width, _ = opencv_image.shape
    # Display image.
    col1.image(opencv_image, channels="BGR")

if post_disaster_file is not None:
    # Convert the file to an opencv image.
    post_bytes = post_disaster_file.read()
    file_bytes = np.asarray(bytearray(post_bytes), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Display image.
    col2.image(opencv_image, channels="BGR")

# Tab2 - FROM URL
col3, col4 = tab2.columns(2)

pre_disaster_url = col3.text_input("Enter the URL of the pre-disaster image")
post_disaster_url = col4.text_input("Enter the URL of the post-disaster image")

if pre_disaster_url:
    file_response = requests.get(pre_disaster_url)
    col3.image(file_response.content, channels="BGR")

if post_disaster_url:
    file_response = requests.get(post_disaster_url)
    col4.image(file_response.content, channels="BGR")

# Prédiction
if col1.button("Predict from file"):
    # Send a POST request to the API
    files = [('files', pre_bytes), ('files', post_bytes)]
    
    response = requests.post(st.secrets['PRED_URL'], files=files)
    
    if response.status_code == 200:
        col5, col6= tab1.columns(2)
        col5.success("Prediction")
        mask = resized_response(response.content, pc_width, pc_height)
        col5.image(mask, channels="BGR")

        m_resp = requests.post(st.secrets['PRED_MASK'], files=files)
        if m_resp.status_code == 200:
            col6.success("Mask")
            mask = resized_response(m_resp.content, pc_width, pc_height)
            col6.image(mask, channels="BGR")
            col9, col10, col11 = tab1.columns(3)
            col10.write("Labels:")
            col10.dataframe(df, hide_index=True)
        else:
            col6.error("Mask cannot be loaded.")
    else:
        tab1.error("Prediction failed.")
        # with open(path.join("logs", "response.txt"), "w") as f:
        #     f.write(str(response.status_code) + "\n\n" + response.text)
        tab1.error(f"Error: {response.status_code} - {response.text}")

if tab2.button("Predict from URL"):
    pre_response = requests.get(pre_disaster_url)
    pre_disaster_image_bytes = pre_response.content
    post_response = requests.get(post_disaster_url)
    post_disaster_image_bytes = post_response.content
    height, width = get_byte_img_shape(pre_response.content)

    # Send a POST request to the API
    files = [('files', pre_disaster_image_bytes), ('files', post_disaster_image_bytes)]
    
    response = requests.post(st.secrets['PRED_URL'], files=files)
    
    if response.status_code == 200:
        col7, col8 = tab2.columns(2)
        col7.success("Prediction")
        mask = resized_response(response.content, width, height)
        col7.image(mask, channels="BGR")

        m_resp = requests.post(st.secrets['PRED_MASK'], files=files)
        if m_resp.status_code == 200:
            col8.success("Mask")
            mask = resized_response(m_resp.content, width, height)
            col8.image(mask, channels="BGR")
            col12, col13, col14 = tab2.columns(3)
            col13.write("Labels:")
            col13.dataframe(df, hide_index=True)
        else:
            col8.error("Mask cannot be loaded.")
    else:
        tab2.error("Prediction failed.")
        tab2.error(f"Error: {response.status_code} - {response.text}")

# Tab3 - From single dual image URL
dual_disaster_url = tab3.text_input("Enter the URL of the image")

if dual_disaster_url:
    file_response = requests.get(dual_disaster_url)
    tab3.image(file_response.content, channels="BGR")
    # file_ext = file_response.headers
    col15, col16 = tab3.columns(2)

    file_ext = "." + file_response.headers['Content-Type'].split('/')[-1]

    button_h = col15.button("Split horizontally and predict")
    button_v = col16.button("Split vertically and predict")

    if button_h:
        files, height, width, pre, post = get_img_chara(file_response.content, file_ext, 'h')
        col15.image(pre, channels="BGR")
        col16.image(post, channels="BGR")
    
        response = requests.post(st.secrets['PRED_URL'], files=files)
    
        if response.status_code == 200:
            col17, col18 = tab3.columns(2)
            col17.success("Prediction")
            mask = resized_response(response.content, width, height)
            col17.image(mask)

            m_resp = requests.post(st.secrets['PRED_MASK'], files=files)
            if m_resp.status_code == 200:
                col18.success("Mask")
                mask = resized_response(m_resp.content, width, height)
                col18.image(mask)
                col19, col20, col21 = tab3.columns(3)
                col20.write("Labels:")
                col20.dataframe(df, hide_index=True)
            else:
                col18.error("Mask cannot be loaded.")
        else:
            tab3.error("Prediction failed.")
            tab3.error(f"Error: {response.status_code} - {response.text}")

    if button_v:
        files, height, width, pre, post = get_img_chara(file_response.content, file_ext, 'v')
        col15.image(pre, channels="BGR")
        col16.image(post, channels="BGR")
    
        response = requests.post(st.secrets['PRED_URL'], files=files)
    
        if response.status_code == 200:
            col17, col18 = tab3.columns(2)
            col17.success("Prediction")
            mask_cv = resized_response(response.content, width, height)
            col17.image(mask_cv, channels="BGR")

            m_resp = requests.post(st.secrets['PRED_MASK'], files=files)
            if m_resp.status_code == 200:
                col18.success("Mask")
                mask_cv = resized_response(m_resp.content, width, height)
                col18.image(mask_cv, channels="BGR")
                col19, col20, col21 = tab3.columns(3)
                col20.write("Labels:")
                col20.dataframe(df, hide_index=True)
            else:
                col18.error("Mask cannot be loaded.")
        else:
            tab3.error("Prediction failed.")
            tab3.error(f"Error: {response.status_code} - {response.text}")
