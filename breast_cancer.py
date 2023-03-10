#import packages
import streamlit as st
import pandas as pd
import plotly_express as px
import numpy as np  
from prediction import predict, predict_proba
from PIL import Image
from streamlit.commands.page_config import Layout
from catboost import CatBoostClassifier, Pool
import random


#----------------------------#
# Upgrade streamlit library
# pip install --upgrade streamlit

#-----------------------------#
# Page layout
icon = Image.open('images/logo.ico')

st.set_page_config(page_title='Breast Cancer Predictions',
                   page_icon=icon,
                   layout='wide',
                   initial_sidebar_state="auto",
                   menu_items=None)

image = Image.open('images/breast_cancer_screen.jpg')


st.title('Breast Cancer Prediction')
st.markdown(' Model to classify breast cancer into benign or malignant')

st.image(image, width=1400)

rand_value = st.checkbox('Random Values')
if rand_value:
  st.header('Cell Features')
  col1, col2, col3 = st.columns(3)
  with col1:
      st.text('Mean Characteristics')
      radius_mean = st.slider('Radius Mean', 6.0, 30.0, step=0.5, value=random.uniform(6.0, 30.0))
      texture_mean = st.slider('Texture Mean', 9.0, 40.0, step=0.5, value=random.uniform(9.0, 40.0))
      perimeter_mean = st.slider('Perimeter Mean', 43, 190, step=1, value=random.randint(43, 190))
      area_mean = st.slider('Area Mean', 143, 2550, step=50, value=random.randint(143, 2550))
      smoothness_mean = st.slider('Smoothness Mean', 0.05, .20, step=0.05, value=random.uniform(0.05, 0.20))
      compactness_mean = st.slider('Compactness Mean', 0.00, 0.35, step=0.05, value=random.uniform(0.00, 0.35))
      concavity_mean = st.slider('Concavity Mean', 0.00, 0.45, step=0.05, value=random.uniform(0.00, 0.45))
      concave_points_mean = st.slider('Concave Points Mean', 0.00, 0.20, step=0.05, value=random.uniform(0.00, 0.20))
      symmetry_mean = st.slider('Symmetry Mean', 0.1, 0.30, step=0.05, value=random.uniform(0.1, 0.30))
      fractal_dimension_mean = st.slider('Fractal_dimension Mean', 0.04, 0.10, step=0.01, value=random.uniform(0.04, 0.10))

  with col2:
      st.text('Standard Error Characteristics')
      radius_se = st.slider('Radius Standard Error', 0.1, 3.0, step=0.25, value=random.uniform(0.1, 3.0))
      texture_se = st.slider('Texture Standard Error', 0.3, 5.0, step=0.50, value=random.uniform(0.3, 5.00))
      perimeter_se = st.slider('Perimeter Standard Error', 0.75, 22.0, step=0.5, value=random.uniform(0.75, 22.0))
      area_se = st.slider('Area Standard Error', 6, 546, step=15, value=random.randint(6, 546))
      smoothness_se = st.slider('Smoothness Standard Error', 0.000, 0.03, step=0.001, value=random.uniform(0.000, 0.03))
      compactness_se = st.slider('Compactness Standard Error', 0.000, 0.12, step=0.005, value=random.uniform(0.000, 0.12))
      concavity_se = st.slider('Concavity Standard Error', 0.000, 0.40, step=0.05, value=random.uniform(0.000, 0.40))
      concave_points_se = st.slider('Concave Points Standard Error', 0.000, 0.055, step=0.005, value=random.uniform(0.000, 0.055))
      symmetry_se = st.slider('Symmetry Standard Error', 0.0070, 0.08, step=0.005, value=random.uniform(0.007, 0.08))
      fractal_dimension_se = st.slider('Fractal Dimension Standard Error', 0.000, 0.03, step=0.001, value=random.uniform(0.000, 0.03))

  with col3:
      st.text('Worst Characteristics')
      radius_worst = st.slider('Radius Worst Value', 7.5, 36.5, step=1.0, value=random.uniform(7.5, 36.5))
      texture_worst = st.slider('Texture Worst Value', 12.0, 50.0, step=2.0, value=random.uniform(12.0, 50.0))
      perimeter_worst = st.slider('Perimeter Worst Value', 50, 256, step=5, value=random.randint(50, 256))
      area_worst = st.slider('Area Worst Value', 185, 4255, step=75, value=random.randint(185, 4225))
      smoothness_worst = st.slider('Smoothness Worst Value', 0.07, 0.25, step=0.01, value=random.uniform(0.07, 0.25))
      compactness_worst = st.slider('Compactness Worst Value', 0.025, 1.10, step=0.05, value=random.uniform(0.025, 1.10))
      concavity_worst = st.slider('Concavity Worst Value', 0.000, 1.25, step=0.05, value=random.uniform(0.000, 1.25))
      concave_points_worst = st.slider('Concave Points Worst Value', 0.000, 0.30, step=0.005, value=random.uniform(0.000, 0.30))
      symmetry_worst = st.slider('Symmetry Worst Value', 0.15, 0.70, step=0.10, value=random.uniform(0.15, 0.70))
      fractal_dimension_worst = st.slider('Fractal Dimension Worst Value', 0.0055, 0.20, step=0.05, value=random.uniform(0.0055, 0.20))



  if st.button('Predict Type of Cancer'):
      result = predict(np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
         concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
         compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
         perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
         symmetry_worst, fractal_dimension_worst]]))
      proba = predict_proba(np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
         concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
         compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
         perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
         symmetry_worst, fractal_dimension_worst]])).tolist()[0]
      if result[0]==0:
          st.success('Cancer is Benign')
      else: 
          st.error('Cancer is Malignant')
      st.text(proba[1]*100)

  with st.expander('Explanation'):
      st.text('Probability of malignant cancer')
      
else:
  st.header('Cell Features')
  col1, col2, col3 = st.columns(3)
  with col1:
      st.text('Mean Characteristics')
      radius_mean = st.slider('Radius Mean', 6.0, 30.0, step=0.5)
      texture_mean = st.slider('Texture Mean', 9.0, 40.0, step=0.5)
      perimeter_mean = st.slider('Perimeter Mean', 43, 190, step=1)
      area_mean = st.slider('Area Mean', 143, 2550, step=50)
      smoothness_mean = st.slider('Smoothness Mean', 0.05, .20, step=0.05)
      compactness_mean = st.slider('Compactness Mean', 0.00, 0.35, step=0.05)
      concavity_mean = st.slider('Concavity Mean', 0.00, 0.45, step=0.05)
      concave_points_mean = st.slider('Concave Points Mean', 0.00, 0.20, step=0.05)
      symmetry_mean = st.slider('Symmetry Mean', 0.1, 0.30, step=0.05)
      fractal_dimension_mean = st.slider('Fractal_dimension Mean', 0.04, 0.10, step=0.01)

  with col2:
      st.text('Standard Error Characteristics')
      radius_se = st.slider('Radius Standard Error', 0.1, 3.0, step=0.25)
      texture_se = st.slider('Texture Standard Error', 0.3, 5.0, step=0.50)
      perimeter_se = st.slider('Perimeter Standard Error', 0.75, 22.0, step=0.5)
      area_se = st.slider('Area Standard Error', 6, 546, step=15)
      smoothness_se = st.slider('Smoothness Standard Error', 0.000, 0.03, step=0.001)
      compactness_se = st.slider('Compactness Standard Error', 0.000, 0.12, step=0.005)
      concavity_se = st.slider('Concavity Standard Error', 0.000, 0.40, step=0.05)
      concave_points_se = st.slider('Concave Points Standard Error', 0.000, 0.055, step=0.005)
      symmetry_se = st.slider('Symmetry Standard Error', 0.0070, 0.08, step=0.005)
      fractal_dimension_se = st.slider('Fractal Dimension Standard Error', 0.000, 0.03, step=0.001)

  with col3:
      st.text('Worst Characteristics')
      radius_worst = st.slider('Radius Worst Value', 7.5, 36.5, step=1.0)
      texture_worst = st.slider('Texture Worst Value', 12.0, 50.0, step=2.0)
      perimeter_worst = st.slider('Perimeter Worst Value', 50, 256, step=5)
      area_worst = st.slider('Area Worst Value', 185, 4255, step=75)
      smoothness_worst = st.slider('Smoothness Worst Value', 0.07, 0.25, step=0.01)
      compactness_worst = st.slider('Compactness Worst Value', 0.025, 1.10, step=0.05)
      concavity_worst = st.slider('Concavity Worst Value', 0.000, 1.25, step=0.05)
      concave_points_worst = st.slider('Concave Points Worst Value', 0.000, 0.30, step=0.005)
      symmetry_worst = st.slider('Symmetry Worst Value', 0.15, 0.70, step=0.10)
      fractal_dimension_worst = st.slider('Fractal Dimension Worst Value', 0.0055, 0.20, step=0.05)



  if st.button('Predict Type of Cancer'):
      result = predict(np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
         concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
         compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
         perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
         symmetry_worst, fractal_dimension_worst]]))
      proba = predict_proba(np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
         concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
         compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
         perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
         symmetry_worst, fractal_dimension_worst]])).tolist()[0]
      if result[0]==0:
          st.success('Cancer is Benign')
      else: 
          st.error('Cancer is Malignant')
      st.text(proba[1]*100)

  with st.expander('Explanation'):
      st.text('Probability of malignant cancer')

