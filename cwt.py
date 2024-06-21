import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
file_path = r'dataverse_files/spectra-1.csv'
df = pd.read_csv(file_path)
print(df.shape)

wavelength = df.columns[1:].to_numpy()
wavelength = np.array([float(item[:-3]) if item.endswith('_nm') else item for item in wavelength])
absorbance = df.values[:][0:]
absorbance = np.delete(absorbance, 0, axis=1)
absorbance = np.vectorize(lambda x: float(x.replace(',', '.')))(absorbance)
print(wavelength[0], wavelength[-1])

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