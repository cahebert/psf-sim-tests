import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from colors import color_scheme as colors
plt.style.use('./paper.mplstyle')
showsig = True

f, a = plt.subplots(2, 1, figsize=(3.35, 3.75), sharex=True)

grey = '#dde3eb'

with open(f'../data/dxi-i-full-radec-03-piff-02.json', 'r') as f:
    data = json.load(f)

for y10, ax in zip([False, True], a):
    # if not y10:
    for type, color, marker, label in zip(
        ['all', 'no_w', 'alpha'],
        [colors.g, colors.y, colors.p],
        ['o','D', 's'],
        [r'all terms', r'$\delta e^{(i)}\delta e^{(j)}$', r'$e^{(i)}x$']
        ):
        if y10:
            subset = data['y10'][type]
            xip = np.array(subset['xip'])
            theta = np.array(subset['r'])
            sig = np.sqrt(subset['xip_var'])
        else:
            subset = data['y1'][type]
            xip = np.mean(subset['xip'], axis=0)
            theta = np.array(subset['r'])
            sig = np.sqrt(subset['xip_var'])

        # fourth order
        ax.plot(theta, xip, '-', color=color, lw=1, zorder=4)
        ax.plot(theta, -xip, ':', color=color, lw=1, zorder=4)
        ax.errorbar(
            theta[xip>0],
            xip[xip>0],
            yerr=sig[xip>0],
            fmt=marker, color=color,
            ms=3, zorder=4, capsize=1.5, capthick=0.5,
            label=label
        )
        ax.errorbar(
            theta[xip<0],
            -xip[xip<0],
            yerr=sig[xip<0],
            fmt=marker, mfc='none', mew=0.75,
            color=color, ms=3, zorder=4, capsize=1.5, capthick=0.5
        )

    if showsig:
        covfile = f'../data/thps_cov_test_matrix_{"Y10" if y10 else "Y1"}.dat'
        constraint = pd.read_csv(covfile, delimiter='\t', header=None).to_numpy()
        covmin = np.sqrt(np.diag(constraint)[:25])  # first half is xi+

        for fraction, ls in zip([1, 0.3],['-', '--']):
            sigma = covmin * fraction
            ax.plot(theta, sigma, ls=ls, lw=2, color=grey, zorder=2)
            ax.fill_between(theta, 0, sigma, color=grey, alpha=0.2, zorder=1)

a[1].text(0.1, 0.85, f'Y10', fontsize=10, transform=a[1].transAxes)
a[0].text(0.1, 0.125, f'Y1', fontsize=10, transform=a[0].transAxes)

from matplotlib.lines import Line2D
sig_labels = [r'$\sigma_{\xi_+}$', r'$0.3\sigma_{\xi_+}$']
handles = [
    Line2D([0], [0], color=grey, lw=2, ls=ls, label=lab)
    for ls, lab in zip(['-', '--'], sig_labels)
]
a[0].legend(handles=handles, loc='upper right', borderaxespad=0.5)
a[1].legend(loc='upper right', borderaxespad=0.5)

a[1].set_xscale('log')
a[1].set_xlabel(r'$\theta$ (arcmin)')
a[1].set_xlim(.5, 200)

# import pickle
# with open('../data/xi-mcal-e460-edges-wldb-varsize-gauss.p', 'rb') as f:
#     xi_calc = pickle.load(f)
#     xi_calc = xi_calc[np.max(list(xi_calc.keys()))]

# area_lsst = 19600  # square degrees
# area_xi = 1.07  # square degrees
# area_ratio = area_xi / area_lsst

# # y1 data is 10x shallower, so multiply by sqrt(10)
# adjustment_factor = area_ratio * 10

# a[0].plot(
#     xi_calc.meanr[xi_calc.meanr<2],
#     np.sqrt(xi_calc.varxip * adjustment_factor)[xi_calc.meanr<2],
#     ls='--',
#     color=colors.p,
#     lw=2,
#     zorder=2,
# )

for ax in a:
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel(r'$\delta\xi_+^{\rm PSF}(\theta)$')
    ax.set_ylim(bottom=4e-10, top=1e-5)

plt.subplots_adjust(hspace=0)

plt.savefig('../figures/dxi-i-full-radec-03-piff-02-2.jpg', dpi=300)
plt.show()