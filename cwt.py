import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from skimage.transform import resize
import cv2
import os
import random
import matplotlib.image as mpimg

file_path = r'dataverse_files/spectra-1.csv'
df = pd.read_csv(file_path)
print(df.shape)

wavelength = df.columns[1:].to_numpy()
wavelength = np.array([float(item[:-3]) if item.endswith('_nm') else item for item in wavelength])
absorbance = df.values[:][0:]
absorbance = np.delete(absorbance, 0, axis=1)
absorbance = np.vectorize(lambda x: float(x.replace(',', '.')))(absorbance)

coef, freqs = pywt.cwt(absorbance[0], np.arange(1,129), 'gaus1')

W, S = np.meshgrid(wavelength, np.arange(1,129))

W_flat = W.flatten()
S_flat = S.flatten()
coef_flat = coef.flatten()

power_spectrum = np.abs(coef) ** 2

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(S_flat, W_flat, coef_flat, c=coef_flat, cmap='inferno', marker='o', alpha=0.7)
ax.set_title('Wavelet Coefficients')
ax.set_ylabel('Wavelength (nm)')
ax.set_xlabel('Scale')
ax.set_zlabel('Wavelet Coefficient')
cbar = plt.colorbar(ax.scatter(S_flat, W_flat, coef_flat, c=coef_flat, cmap='inferno'))
cbar.set_label('Coefficient Value')

# Create the spectrogram plot
ax2 = fig.add_subplot(122)
pcm = ax2.pcolormesh(wavelength, np.arange(1, 129), coef, cmap='inferno')  # Use a suitable colormap
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Scale')
ax2.set_title('2D Spectrogram')
cbar2 = fig.colorbar(pcm, ax=ax2, label='Coefficient Value', cmap='inferno')
ax2.set_aspect('auto')

plt.tight_layout()  
plt.show()

# print(coef)

resized_coef = cv2.resize(coef, (256, 256))
normalized_coef = ((resized_coef - np.min(resized_coef)) / (np.max(resized_coef) - np.min(resized_coef))) * 255
rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
rgb_image[:,:,0] = normalized_coef  # Red channel
rgb_image[:,:,1] = normalized_coef  # Green channel
rgb_image[:,:,2] = normalized_coef  # Blue channel

output_folder = 'absorbance_images'
os.makedirs(output_folder, exist_ok=True)

for i in range(len(absorbance)):
    coef, freqs = pywt.cwt(absorbance[i], np.arange(1, 129), 'gaus1')
    resized_coef = cv2.resize(coef, (256, 256))
    normalized_coef = ((resized_coef - np.min(resized_coef)) / (np.max(resized_coef) - np.min(resized_coef))) * 255
    rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = normalized_coef  # Red channel
    rgb_image[:, :, 1] = normalized_coef  # Green channel
    rgb_image[:, :, 2] = normalized_coef  # Blue channel

    image_path = os.path.join(output_folder, f'absorbance_{i}.png')
    cv2.imwrite(image_path, rgb_image)
    print(f'Image {i} saved successfully at {image_path}')

folder_path = 'absorbance_images'

# Get a list of file paths for all images in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]

# Randomly pick 10 images
selected_images = random.sample(image_files, 10)

# Plot the selected images
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, image_path in enumerate(selected_images):
    ax = axes[i // 5, i % 5]
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_title(os.path.basename(image_path))
    ax.axis('off')

plt.tight_layout()
plt.show()