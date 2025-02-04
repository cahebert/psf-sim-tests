import matplotlib.pyplot as plt
import numpy as np
import fitsio
import pandas as pd
plt.style.use('./paper.mplstyle')  # change this to a paper specific one

def load_catalog(catpath):
    catalog = fitsio.read(catpath)
    catalog = catalog[~catalog['flagged']]

    cat = pd.DataFrame(catalog)

    cat['dtt'] = (cat['T'] - cat['psf_T']) / cat['T']
    cat['dg1'] = (cat['g1'] - cat['psf_g1'])
    cat['dg2'] = (cat['g2'] - cat['psf_g2'])

    cat['dtt4'] = (cat['T4'] - cat['psf_T4']) / cat['T4']
    cat['de4_1'] = (cat['e4_1'] - cat['psf_e4_1'])
    cat['de4_2'] = (cat['e4_2'] - cat['psf_e4_2'])

    return cat

cat = load_catalog('/Users/clairealice/Documents/shear/testing-piff/data/cat-i-full-radec-03-piff-01.fits')
print(np.mean(cat['dtt']))

f, a = plt.subplots(3, 2, sharey='row', figsize=(3.35, 5))#, gridspec_kw={'wspace':0.05})

lw = 1.25
c2 = '#1D5D4C'
c4 = '#7D7C84'
ls1 = '-'
ls2 = '--'

Tbins = np.linspace(0, 1, 50)
a[0,0].hist(cat['T'], Tbins, histtype='step', color=c2, lw=lw, label=r'$T^{(2)}$')
# a[0,0].hist(cat['T4', Tbins, histtype='step', color=c4, lw=lw, label=r'$T^{(4)}$')

e2bins = np.linspace(-0.1, 0.1, 50)
a[1,0].hist(cat['g1'], e2bins, histtype='step', color=c2, ls=ls1, lw=lw, label=r'$e^{(2)}_1$')
a[1,0].hist(cat['g2'], e2bins, histtype='step', color=c2, ls=ls2, lw=lw, label=r'$e^{(2)}_2$')

# e4bins = np.linspace(np.min([cat['e4_1'], cat['e4_2']]), np.max([cat['e4_1'], cat['e4_2']]), 50)
# a[2,0].hist(cat['e4_1'], e4bins, histtype='step', color=c4, ls=ls1, lw=lw, label=r'$e^{(4)}_1$')
# a[2,0].hist(cat['e4_2'], e4bins, histtype='step', color=c4, ls=ls2, lw=lw, label=r'$e^{(4)}_2$')

reserved_cat = cat[cat['reserved']]
dTbins = np.linspace(-0.1, 0.1, 50)
a[0,1].hist(reserved_cat['dtt'], dTbins, histtype='step', color=c2, lw=lw, label=r'$\frac{\delta T^{(2)}}{T^{(2)}}$')
# a[0,1].hist(cat['dtt4'], dTbins, histtype='step', color=c4, lw=lw, label=r'$\frac{\delta T^{(4)}}{T^{(4)}}$')

de2bins = np.linspace(-0.1, 0.1, 50)
a[1,1].hist(reserved_cat['dg1'], de2bins, histtype='step', color=c2, ls=ls1, lw=lw, label=r'$e^{(2)}_1$')
a[1,1].hist(reserved_cat['dg2'], de2bins, histtype='step', color=c2, ls=ls2, lw=lw, label=r'$e^{(2)}_2$')

# de4bins = np.linspace(np.min([cat['de4_1'], cat['de4_2']]), np.max([cat['de4_1'], cat['de4_2']]), 50)
# a[2,1].hist(cat['de4_1'], de4bins, histtype='step', color=c4, ls=ls1, lw=lw, label=r'$e^{(4)}_1$')
# a[2,1].hist(cat['de4_2'], de4bins, histtype='step', color=c4, ls=ls2, lw=lw, label=r'$e^{(4)}_1$')

for ax in a.flatten()[1:]:
    ax.axvline(0, ls='--', color='lightgrey', alpha=0.8, zorder=2)

for ax in [a[0,0], a[1,0], a[2,0]]:
    ax.legend(borderaxespad=0.5)

a[0,0].set_xlabel(r'$T$ (arcsec$^2$)')
a[0,1].set_xlabel(r'$\delta T / T$')

a[1,0].set_xlabel(r'$e^{(2)}$')
a[1,1].set_xlabel(r'$\delta e^{(2)}$')

a[2,0].set_xlabel(r'$e^{(4)}$')
a[2,1].set_xlabel(r'$\delta e^{(4)}$')

# from matplotlib.lines import Line2D
# sig_labels = [r'$0.1\sigma_+^{\rm min}$',
#               r'$0.2\sigma_+^{\rm min}$',
#               r'$0.3\sigma_+^{\rm min}$']
# handles = [Line2D([0], [0], color='#dde3eb', lw=2, ls=ls, label=lab)
#             for ls, lab in zip(['-', '--', ':'], sig_labels)]
# a[1].legend(handles=handles, loc='upper right', borderaxespad=0.75)

# plt.savefig('../figures/dxi_y1y10.jpg', dpi=300)
plt.show()