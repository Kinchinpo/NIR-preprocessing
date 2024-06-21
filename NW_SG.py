import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from chemotools.derivative import NorrisWilliams, SavitzkyGolay
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
file_path = r'dataverse_files/spectra-1.csv'
df = pd.read_csv(file_path)
print(df.shape)

wavelength = df.columns[1:].to_numpy()
wavelength = np.array([float(item[:-3]) if item.endswith('_nm') else item for item in wavelength])
absorbance = df.values[:][0:]
absorbance = np.delete(absorbance, 0, axis=1)
absorbance = np.vectorize(lambda x: float(x.replace(',', '.')))(absorbance)

sg_deri1_data = signal.savgol_filter(absorbance, window_length=7, polyorder=3, deriv=2, mode='nearest')
nw = NorrisWilliams(window_size=7, gap_size=3, derivative_order=2)
nw_deri1_data = nw.fit_transform(absorbance)

for i in range(1):
    sg_data = sg_deri1_data[i]
    nw_data = nw_deri1_data[i]
    plt.plot(wavelength, sg_data, color='b', label=f'1st-order sg')
    plt.plot(wavelength, nw_data, color='g', label=f'1st-order nw')

plt.xlabel('Wavelength')
plt.title('1st-order derivative of SG and NW of one manure sample')
plt.legend()
plt.show()