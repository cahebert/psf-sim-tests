import matplotlib.pyplot as plt
import numpy as np

plt.style.use('./figure_scripts/paper.mplstyle')  # change this to a paper specific one

data = np.loadtxt('./data/tq_tratio_4thmom_012725.csv')  # load the TQ sims
# print(data.shape)

f, a = plt.subplots(2, 1, figsize=(3.35, 3), sharex=True, gridspec_kw={'hspace':0})

x = data[0][::-1]
el = np.searchsorted(x, x[(x>0.6)&(x<2)][0])
er = np.searchsorted(x, x[(x>0.6)&(x<2)][-1])

xpts = np.linspace(0,3)

for ax, coeff_tuple in zip(a, [(data[1][::-1], data[2][::-1]), (data[3][::-1], data[4][::-1])]):
    m = (coeff_tuple[0][er]-coeff_tuple[0][el]) / (x[er] - x[el])
    c = coeff_tuple[0][el+2] - m * (x[el+2] - x[0])
    ax.plot(xpts, xpts * m + c, color='lightgrey', lw=1.5)

    ax.plot(x, coeff_tuple[0], ls='-', lw=1.5, label='real')
    ax.plot(x, coeff_tuple[1], ls=':', lw=1.5, label='complex')

a[0].legend(loc='lower left')

# a[0].set_ylabel(r'$\frac{\Delta c}{\Delta e^{(4)}_{\rm PSF}}$')
# a[1].set_ylabel(r'$\frac{\Delta c}{e^{(4)} \frac{\Delta T^{(4)} }{T^{(4)}}_{\rm PSF}}$')
a[0].set_ylabel(r'$\delta c / \delta e^{(4)}_{\rm PSF}$')
a[1].set_ylabel(r'$\delta c / \left(e^{(4)} \frac{\delta T^{(4)} }{T^{(4)}}\right)_{\rm PSF}$')
a[1].set_xlabel(r'$\frac{T_{\rm PSF}}{T_{\rm gal}}$')

a[1].set_xlim([0,3])
a[0].set_ylim(bottom=-0.1)
a[1].set_ylim(bottom=-0.1)

plt.subplots_adjust(right=0.95, top=0.95, bottom=0.15)

plt.savefig('./figures/tq_tratio_4thmom.jpg', dpi=300)
plt.show()