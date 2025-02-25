import matplotlib.pyplot as plt
import numpy as np
import fitsio
import pandas as pd
import skyproj
plt.style.use('./paper.mplstyle')  # change this to a paper specific one

def load_catalog(catpath):
    catalog = fitsio.read(catpath)
    catalog = catalog[~catalog['flagged']]

    cat = pd.DataFrame(catalog)
    return cat

cat = load_catalog('~/Documents/shear/testing-piff/data/cat-i-full-radec-03-piff-02.fits')
summary = fitsio.read('../data/observing-summary-radec-03.fits')

temp = np.zeros((2, len(cat)))
for v in np.unique(cat['visit']):
    temp[:,cat['visit']==v] = summary[summary['visit']==v]['skybg'], summary[summary['visit']==v]['airmass']

f, a = plt.subplots(2,2, figsize=(3.35,3.5))

for ax, param, label, ticks in zip(
    a.flatten(),
    ['N', 'am_s2n', 1, 0],
    [r'N$_{sources}$', r'$S/N$', 'airmass', 'sky background'],
    [[0,10000,20000],[],[1.1, 1.2],[40000,50000]]):

    sp = skyproj.McBrydeSkyproj(ax=ax, n_grid_lat=3)
    if param=='N':
        sp.draw_hpxbin(
            cat['ra'], cat['dec'],
            zoom=True, xsize=600, cmap='magma')
    elif param == 'am_s2n':
        sp.draw_hpxbin(
            cat['ra'], cat['dec'], C=cat[param],
            zoom=True, xsize=600, cmap='magma')
    else:
        sp.draw_hpxbin(
            cat['ra'], cat['dec'], C=temp[param],
            zoom=True, xsize=600, cmap='magma')

    # ticks=[-tick, 0, tick]
    cb = sp.draw_colorbar(
        label=label,
        pad=0.05,
        ticks=ticks,
        fontsize=9,
        location='top'
        )
    sp.ax._ticklabels_visibility['top']=False

    # bottom x labels
    if param!='N':
        sp.ax.set_xlabel('Right Ascension', fontsize=11)
    else:
        sp.ax.set_xlabel('')
        sp.ax._ticklabels_visibility['bottom']=False
    # right y labels
    if param in [1]:
        sp.ax.set_ylabel('')
        sp.ax._ticklabels_visibility['left']=False
    else:
        sp.ax.set_ylabel('Declination', fontsize=10)

a[-1,1].axis('off')

plt.subplots_adjust(top=0.925, bottom=0.05, right=0.95, left=0.2, wspace=0.1, hspace=0.1)
plt.savefig('../figures/condition-sky-maps.jpg', dpi=300)
plt.show()
