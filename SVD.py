import streamlit as st
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris # pip install scikit-learn
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler



st.write("This App resizes your image based on input")
url = st.text_input("Enter image url",'https://helpx.adobe.com/content/dam/help/en/photoshop/using/convert-color-image-black-white/jcr_content/main-pars/before_and_after/image-after/Landscape-BW.jpg')

image = io.imread(url)[:, :, 0]
print(f'Your image size {image.shape}')
st.image(image, caption="Your image")

U, sing_vals, V = np.linalg.svd(image)

sigma = np.zeros(shape = image.shape)

np.fill_diagonal(sigma, sing_vals)
U.shape, sigma.shape, V.shape

k=int(st.text_input("Enter square matrix size","100"))
trunc_U = U[:, :k]
trunc_sigma = sigma[:k, :k]
trunc_V = V[:k, :]
trunc_U.shape, trunc_sigma.shape, trunc_V.shape

fig, ax = plt.subplots(1, 1, figsize=(15, 7))

ax.imshow(trunc_U@trunc_sigma@trunc_V, cmap = 'gray')
ax.set_title('После SVD')
st.pyplot(fig)
