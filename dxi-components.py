import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from colors import color_scheme as colors
plt.style.use('./paper.mplstyle')
showsig = True

from matplotlib.patches import Wedge
from matplotlib.markers import MarkerStyle
wedge_up = Wedge(center=0, r=0.1, theta1=0, theta2=180)
wedge_lw = Wedge(center=0, r=0.1, theta1=180, theta2=0)
# Convert the patch to a Path
path_up = wedge_up.get_path().transformed(wedge_up.get_transform())
path_lw = wedge_lw.get_path().transformed(wedge_lw.get_transform())
# Wrap the Path into a MarkerStyle
hemisphere_up_marker = MarkerStyle(path_up)
hemisphere_lw_marker = MarkerStyle(path_lw)

def plot_dxi_band(ax, xip, theta, sig, label, params, c=1, alpha_fill=0.2):
    line2dparams = {k:p for k,p in params.items() if 'cap' not in k}
    ax.plot(theta, xip*c, **line2dparams, label=label)
    ax.plot(theta, -xip*c, mfc='none', **line2dparams)

    ax.fill_between(
        theta, abs(xip*c) - sig * abs(c), abs(xip*c) + sig * abs(c),
        alpha=alpha_fill, color=params['color'], zorder=2,
    )
    return ax

f, a = plt.subplots(2, 1, figsize=(3.35, 4.25), sharex=True, sharey=True)

grey = '#dde3eb'

with open(f'../data/dxi-i-full-radec-04-piff-01-cornercut-empirical.json', 'r') as f:
    data = json.load(f)

frac = 0.35
alpha = 0.8

for y10, ax, color_list in zip(
    [False, True], a, [[colors.g, colors.p, colors.y], [colors.g, colors.p, colors.y]],
    ):
    for type, marker, alpha, color, label, zorder in zip(
        ['all', 'w_0', 'w_12'], ['o', hemisphere_up_marker, hemisphere_lw_marker], [0.9, 0.75, 0.75], color_list,
        [r'full', r'$\delta e^{(i)}\delta e^{(j)}$ terms', r'all other terms'], [4,3,2]
        ):
        if y10:
            subset = data['y10'][type]
            xip = np.array(subset['xip']) * frac
            theta = np.array(subset['r'])
            sig = np.sqrt(subset['xip_var']) * frac
        else:
            subset = data['y1'][type]
            xip = np.mean(subset['xip'], axis=0) * frac
            theta = np.array(subset['r'])
            sig = np.sqrt(subset['xip_var']) * frac

        params = dict(
            marker=marker,
            color=color,
            ms=4,
            zorder=zorder,
            lw=0,
            mew=0.75,
            alpha=alpha,
        )
        plot_dxi_band(
            ax, xip, theta, sig,
            label=label, params=params,
            alpha_fill=0.2 if type=='all' else 0.15)

    if showsig:
        covfile = f'../data/thps_cov_test_matrix_V2_{"Y10" if y10 else "Y1"}.dat'
        constraint = pd.read_csv(covfile, delimiter='\t', header=None).to_numpy()
        covmin = np.sqrt(np.diag(constraint)[:25])  # first half is xi+
        theta_sig = np.copy(theta)
        theta_sig[0] = 0.5
        theta_sig[-1] = 50
        for fraction, ls, lab in zip([1, 0.3],['-', '--'], [r'$\sigma_{\xi_+}$', r'$0.3\sigma_{\xi_+}$']):
            sigma = covmin * fraction
            ax.plot(theta, sigma, ls=ls, lw=2, color=grey, zorder=2, label=lab)
            ax.fill_between(theta_sig, 0, sigma, color=grey, alpha=0.2, zorder=1)

        if not y10:
            ylimu = covmin[0] * 2
            ylimb = sigma[-1] / 100

a[1].text(0.7125, 0.8, r'Y10 ($riz$)', fontsize=10, transform=a[1].transAxes)
a[0].text(0.7125, 0.8, r'Y1 ($riz$)', fontsize=10, transform=a[0].transAxes)

h1, l1 = a[1].get_legend_handles_labels()
legend_order = [-2,-1,1,2,0]
a[1].legend(
    handles=[h1[i] for i in legend_order],
    labels=[l1[i] for i in legend_order],
    loc='lower left',
    borderaxespad=0.1, ncol=3, handletextpad=0.2, columnspacing=0.25,
    bbox_to_anchor=(0,1))

a[1].set_xscale('log')
a[-1].set_xlabel(r'$\theta$ (arcmin)')
a[1].set_xlim(np.min(theta)-.05, np.max(theta)+5)

for ax in a:
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel(r'$\delta\xi_+^{\rm PSF}(\theta)$')
    ax.set_ylim(bottom=ylimb, top=ylimu)

plt.subplots_adjust(hspace=0)

plt.savefig(f'../figures/dxi-riz-full-radec-04-piff-01-cornercut-empirical.jpg', dpi=300)
plt.show()