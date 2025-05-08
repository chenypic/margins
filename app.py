# client.py - improved version
import requests
import cv2
import numpy as np
import os
import base64
import argparse


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
os.environ['STREAMLIT_SERVER_PORT'] = '4334'


def send_image(image_path, server_url):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist")
        return
    
    # Open the image file
    with open(image_path, 'rb') as f:
        files = {'image': f}
        
        try:
            # Send the image to the server
            response = requests.post(server_url, files=files)
            
            # Check if request was successful
            if response.status_code == 200:
                # Get JSON data from response
                response_data = response.json()
                
                # Extract image size
                image_size = response_data.get('image_size')
                print(f"Image size: {image_size}")

                # Extract 预测结果
                pred_img = response_data.get('pred_img')
                print(f"预测结果: {pred_img}")
                
                # Extract and decode grayscale image
                img_base64 = response_data.get('grayscale_image')
                if img_base64:
                    # Decode base64 to bytes
                    img_bytes = base64.b64decode(img_base64)
                    
                    # Convert bytes to numpy array
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    gray_img = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
                    
                    if gray_img is not None:
                        # Save grayscale image
                        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_gray.png"
                        cv2.imwrite(output_filename, gray_img)
                        print(f"Grayscale image saved as '{output_filename}'")
                        
                        # Save image size to a text file
                        size_filename = os.path.splitext(os.path.basename(image_path))[0] + "_size.txt"
                        with open(size_filename, 'w') as size_file:
                            size_file.write(f"Width: {image_size['width']}, Height: {image_size['height']}")
                        print(f"Image size saved as '{size_filename}'")
                    else:
                        print("Error decoding grayscale image")
                else:
                    print("No grayscale image received in response")
            else:
                print(f"Error: Server returned status code {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error communicating with server: {e}")

        return pred_img,gray_img




model = joblib.load('xgboost_model.pkl')


##### streamlit 部分******
st.set_page_config(layout="wide")




# Streamlit user interface
st.title("Prediction System of positive margins")




# if uploaded_file is not None:
#     # Read image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Original Image", use_container_width =False)
        



# col1, col2 = st.columns(2)
        
# # First column inputs
# with col1:
#     #st.subheader("Patient Information I")
#     age = st.selectbox("Age (0=<48, 1=>48):", options=[0, 1], 
#                         format_func=lambda x: '<48 (0)' if x == 0 else '>48 (1)')
            
#     HPV = st.selectbox("HPV (0=Negative, 1=Positive):", options=[0, 1], 
#                         format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')
            
#     TCT_HSIL = st.selectbox("TCT_HSIL (0=Negative, 1=Positive):", options=[0, 1], 
#                             format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')
        
#     # Second column inputs
# with col2:
#     #st.subheader("Patient Information II")
#     ECC = st.selectbox("ECC (0=Negative, 1=Positive):", options=[0, 1], 
#                             format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')
            
#     margin = st.selectbox("margin (0=Negative, 1=Positive):", options=[0, 1], 
#                         format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')
            
#     Transformation = st.selectbox("Transformation (0=Negative, 1=Positive):", options=[0, 1], 
#                                 format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')
    


feature_names = [
    "TCT≥HSIL", "THPV16/18", "Evaluation-HSIL", "deepLEEP"
]


HSIL = st.selectbox("TCT≥HSIL (0=Negative, 1=Positive):", options=[0, 1], 
                format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')
            
HPV = st.selectbox("THPV16/18 (0=Negative, 1=Positive):", options=[0, 1], 
                format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')

Evaluation = st.selectbox("Evaluation-HSIL (0=Negative, 1=Positive):", options=[0, 1], 
                format_func=lambda x: 'Negative (0)' if x == 0 else 'Positive (1)')


# Define feature names




uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if st.button("Predict"):

    


    # Display original and prediction images side by side
    img_col1, img_col2 = st.columns(2)
        
    with img_col1:
        if uploaded_file is not None:

            temp_dir = tempfile.gettempdir()
        
            # Create a path for the file in the temp directory
            file_name = os.path.join(temp_dir, uploaded_file.name)
        
            # Save the uploaded file to the temp directory
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getvalue())
        
            # Get the absolute path
            absolute_path = os.path.abspath(file_name)


            pred_img,heat_img = send_image(str(absolute_path), 'http://chendiandian.vip:7807/process_image')
            print('********预测值：',np.float32(pred_img['pred_img']))
            cv2.imwrite('heat_img.png', heat_img)

            feature_values = [HSIL, HPV, Evaluation,np.float32(pred_img['pred_img'])]
            features = np.array([feature_values])

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

            shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)

            image = Image.open(uploaded_file)
            # st.image(image, caption="Original Image", use_container_width=False)
            st.image(image, caption="Original Image", width=400)
            
        
    with img_col2:
            # Load and display the heatmap prediction
        heat_img = Image.open("heat_img.png")
        # st.image(heat_img, caption="Prediction Heatmap", use_container_width=False)
        st.image(heat_img, caption="Prediction Heatmap", width=400)

    

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our predictive model, you have a high risk of developing positive margins, with an estimated probability of {probability:.1f}%.  "
            f"Although this result is an estimate based on the model's calculations, it suggests a significant potential risk. "
            "I strongly recommend that you consult a gynecological specialist as soon as possible for further evaluation, accurate diagnosis, "
            "and timely management or treatment if necessary. "

        )
    else:
        advice = (
            f"According to our predictive model, your risk of developing positive margins is relatively low, with an estimated probability of {probability:.1f}%. "
            f"However, it remains very important to maintain a healthy lifestyle and undergo regular health screenings. "
            "We recommend scheduling periodic check-ups and promptly consulting a doctor if you experience any concerning symptoms. "

        )

    st.write(advice)

    st.image("shap_force_plot.png")

    


# else:
#     # Show only the original image before prediction
#     if uploaded_file is not None:
#         st.image(image, caption="Original Image", use_container_width=False)