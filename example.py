import numpy as np
from effdiff import effdiff
from matplotlib import pyplot as plt

d = np.load('data/example_data.npz')

tr = d['tracer']
rac = d['rac']
dx = d['dx']
dy = d['dy']

N = 100 # resolution of Keff calculation
eng = effdiff.EffDiffEngine(rac,dx,dy,N)
result = eng.calc_Le2(tr)

# plot result
plt.plot(eng.A, result['Le2'])
plt.xlabel(r'A (m$^2$)')
plt.ylabel(r'$L_e^2$ (m$^2$)')
plt.title('Equivalent Length Squared')
plt.show()
