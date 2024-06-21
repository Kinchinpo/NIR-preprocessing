import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
file_path = r'dataverse_files/spectra-1.csv'
df = pd.read_csv(file_path)

wavelength = df.columns[1:].to_numpy()
wavelength = np.array([float(item[:-3]) if item.endswith('_nm') else item for item in wavelength])
absorbance = df.values[:][0:]
absorbance = np.delete(absorbance, 0, axis=1)
absorbance = np.vectorize(lambda x: float(x.replace(',', '.')))(absorbance)

sg_smooth = signal.savgol_filter(absorbance, window_length=7, polyorder=2, deriv=0, mode='nearest')
snv = StandardNormalVariate()
data_snv = snv.fit_transform(sg_smooth)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121)
for i in range(absorbance.shape[0]):
    ax.plot(wavelength, absorbance[i])
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorbance')
ax.set_title('Before preprocessing')

ax2 = fig.add_subplot(122)
for i in range(absorbance.shape[0]):
    ax2.plot(wavelength, data_snv[i])
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Absorbance')
ax2.set_title('After preprocessing (SG smooth & SNV)')

plt.tight_layout()
plt.show()