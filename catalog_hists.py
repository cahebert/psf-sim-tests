import matplotlib.pyplot as plt
import numpy as np
import fitsio
import pandas as pd
plt.style.use('./paper.mplstyle')  # change this to a paper specific one
plt.rcParams['text.usetex'] = False

def load_catalog(catpath):
    catalog = fitsio.read(catpath)
    catalog = catalog[~catalog['flagged']]

    cat = pd.DataFrame(catalog)

    cat['dtt'] = (cat['T'] - cat['psf_T']) / cat['psf_T']
    cat['dg1'] = (cat['g1'] - cat['psf_g1'])
    cat['dg2'] = (cat['g2'] - cat['psf_g2'])

    cat['dtt4'] = (cat['T4'] - cat['psf_T4']) / cat['psf_T4']
    cat['de4_1'] = (cat['e4_1'] - cat['psf_e4_1'])
    cat['de4_2'] = (cat['e4_2'] - cat['psf_e4_2'])

    return cat

cat = load_catalog('~/workarea/piff-catalogs/cat-i-full-radec-03-piff-02.fits')
print(np.mean(cat['dtt']))

cat_res = cat[cat['reserved']]
cat = cat[~cat['reserved']]

f, a = plt.subplots(4,2, sharey=True, figsize=(3.35, 5.5), gridspec_kw={'wspace':0.075, 'hspace':0.55})

nbins = 51
lw = 1
lw1 = 0.8
lw2 = 0.8

for catalog, color, label, ls in zip([cat, cat_res], ['#D60270', '#0038A8'], ['PSF stars', 'reserve stars'], ['-','--']):
    n_total = len(catalog)
    weights = np.ones(n_total)/n_total
    
    catalog['dtt'] = (catalog['T'] - catalog['psf_T']) / catalog['psf_T']
    catalog['dg1'] = (catalog['g1'] - catalog['psf_g1'])
    catalog['dg2'] = (catalog['g2'] - catalog['psf_g2'])

    catalog['dtt4'] = (catalog['T4'] - catalog['psf_T4']) / catalog['psf_T4']
    catalog['de4_1'] = (catalog['e4_1'] - catalog['psf_e4_1'])
    catalog['de4_2'] = (catalog['e4_2'] - catalog['psf_e4_2'])
    
    T2bins = np.linspace(0, 0.7, int(nbins*2/3))
    a[0,0].hist(catalog['T'], T2bins, histtype='step', color=color, ls=ls, lw=lw, label=label, weights=weights)
    dT2bins = np.linspace(-0.1, 0.1, nbins)
    a[0,1].hist(catalog['dtt'], dT2bins, histtype='step', color=color, ls=ls, lw=lw, weights=weights)
    
    T4bins = np.linspace(0, 0.075, int(nbins*2/3))
    a[1,0].hist(catalog['T4'], T4bins, histtype='step', color=color, ls=ls, lw=lw, weights=weights)
    dT4bins = np.linspace(-0.5, 0.5, nbins)
    a[1,1].hist(catalog['dtt4'], dT4bins, histtype='step', color=color, ls=ls, lw=lw, weights=weights)

    ebins = np.linspace(-0.06, 0.06, nbins)
    debins = np.linspace(-0.06, 0.06, nbins)
    a[2,0].hist(catalog['g1'], ebins, histtype='step', color=color, ls=ls, lw=lw1, weights=weights, label=r'$g^{(2)}_1$')
    a[2,0].hist(catalog['g2'], ebins, alpha=0.15, color=color, ls=ls, label=r'$g^{(2)}_2$', weights=weights)
    a[2,0].hist(catalog['g2'], ebins, histtype='step', alpha=0.25, color=color, ls=ls, lw=lw2, weights=weights)

    a[2,1].hist(catalog['dg1'], debins, histtype='step', color=color, lw=lw1, ls=ls, weights=weights, label=r'$g^{(2)}_1$')
    a[2,1].hist(catalog['dg2'], debins, alpha=0.15, color=color, ls=ls, weights=weights)
    a[2,1].hist(catalog['dg2'], debins, histtype='step', alpha=0.25, color=color, ls=ls, lw=lw2, weights=weights)
    
    a[3,0].hist(catalog['e4_1'], ebins, histtype='step', color=color, lw=lw1, ls=ls, weights=weights, label=r'$e^{(4)}_1$')
    a[3,0].hist(catalog['e4_2'], ebins, alpha=0.15, color=color, ls=ls, label=r'$e^{(4)}_2$', weights=weights)
    a[3,0].hist(catalog['e4_2'], ebins, histtype='step', alpha=0.25, color=color, ls=ls, lw=lw2, weights=weights)
    
    a[3,1].hist(catalog['de4_1'], debins, histtype='step', color=color, lw=lw1, ls=ls, weights=weights, label=r'$e^{(4)}_1$')
    a[3,1].hist(catalog['de4_2'], debins, alpha=0.15, color=color, ls=ls, weights=weights)
    a[3,1].hist(catalog['de4_2'], debins, histtype='step', alpha=0.25, color=color, ls=ls, lw=lw2, weights=weights)

a[0,0].text(0.95,0.825, 'PSF stars', color='#D60270', horizontalalignment='right', transform=a[0,0].transAxes)
a[0,0].text(0.95,0.675, 'reserve stars', color='#0038A8', horizontalalignment='right', transform=a[0,0].transAxes)

from matplotlib.patches import Patch

handles = [Patch(edgecolor='dimgrey', facecolor='none', lw=lw1, label=r'$g_1$'), 
           Patch(color='dimgrey',lw=lw2, alpha=0.2, label=r'$g_2$')]
a[2,0].legend(handles=handles, borderaxespad=0.25, loc='upper left')
handles = [Patch(edgecolor='dimgrey', facecolor='none', lw=lw1, label=r'$e_1$'), 
           Patch(color='dimgrey',lw=lw2, alpha=0.2, label=r'$e_2$')]
a[3,0].legend(handles=handles, borderaxespad=0.25, loc='upper left')

for ax in list(a[:,1].flatten())+list(a[2:,0].flatten()):
    ax.axvline(0, ls='-', color='lightgrey', alpha=0.8, zorder=1)

a[0,0].set_xlabel(r'$T^{(2)}$ (arcsec$^2$)')
a[0,1].set_xlabel(r'$\delta T^{(2)} / T^{(2)}$')

a[1,0].set_xlabel(r'$T^{(4)}$ (arcsec$^2$)')
a[1,1].set_xlabel(r'$\delta T^{(4)} / T^{(4)}$')

a[2,0].set_xlabel(r'$g^{(2)}$')
a[2,1].set_xlabel(r'$\delta g^{(2)}$')

a[3,0].set_xlabel(r'$e^{(4)}$')
a[3,1].set_xlabel(r'$\delta e^{(4)}$')

plt.subplots_adjust(top=0.975, bottom=0.085, right=0.925)
plt.savefig('../figures/cat_hist.jpg', dpi=300)
# plt.show()
