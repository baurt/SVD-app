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




url = st.text_input("Enter image url","John Doe")

image = io.imread(url)[:, :, 0]
print(f'Your image size {image.shape}')
plt.imshow(image, cmap='gray')

U, sing_vals, V = np.linalg.svd(image)

sigma = np.zeros(shape = image.shape)

np.fill_diagonal(sigma, sing_vals)
U.shape, sigma.shape, V.shape

k=int(st.text_input("Enter square matrix size","number"))
trunc_U = U[:, :k]
trunc_sigma = sigma[:k, :k]
trunc_V = V[:k, :]
trunc_U.shape, trunc_sigma.shape, trunc_V.shape

fig, ax = plt.subplots(1, 2, figsize=(15, 7))


ax[0].imshow(U@sigma@V, cmap = 'gray')
ax[0].set_title('Исходное')
ax[1].imshow(trunc_U@trunc_sigma@trunc_V, cmap = 'gray')
ax[1].set_title('После SVD')
