import matplotlib.pyplot as plt
plt.style.use('~/Documents/clem.mplstyle')
from colors import color_scheme as colors
import json
import numpy as np
import scipy.interpolate
import scipy.integrate

def plot_tratio(ax, c1, c2, nbins=40):
    import fitsio
    bins = np.linspace(0, 2, nbins+1)
    lw=1.5
    out = {}
    for nepoch, ls, c, ec, label in zip(
        [460, 46],
        ['-', '-'],
        [c1, c2],
        [c1, c2],
        ['Y10', 'Y1']
    ):
        data = fitsio.read(f'/Users/clairealice/Documents/git/testing-piff-paper/data/summary-mcal-e{nepoch}-edges-wldb-varsize-gauss.fits')

        # if nepoch == 46:
        # ax.hist(
        #     data['T_ratio'],
        #     bins=bins,
        #     # histtype='step',
        #     color=c,
        #     # density=True,
        #     alpha=0.25 if nepoch==46 else 0.05,
        #     zorder=-1
        #     )
        ax.hist(
            data['T_ratio'],
            bins=bins,
            color=c,
            histtype='step',
            ls=ls,
            lw=lw,
            # alpha=0.8,
            label=label,
            zorder=-1
            )
        h = np.histogram(
            data['T_ratio'],
            bins=bins,
            density=True
            )
        out[nepoch]=h
        # ax.text(2.05, 1400 if nepoch==460 else 500, label, color=c, fontsize=9)
        ax.text(0.35, 2700 if nepoch==460 else 1100, label, color=c, fontsize=9)

    # ax.legend(loc='center right', borderaxespad=0.2)
    # ax.set_ylabel(r'N$_{\rm gal}$')

    return ax, out

if __name__ == '__main__':
    with open('moment_response_tratio_results.json', 'r') as f:
        responses = json.load(f)
    with open('moment_response_tratio_results_mcal.json', 'r') as f:
        mcal_responses = json.load(f)
    # with open('moment_response_images.npy', 'rb') as f:
    imgs = np.load('test_images.npy')
    size_ratio = np.array(responses['size_ratio'])
    coeffs = {
        'lssty1':{'alpha2':0, 'alpha4':0},
        'lssty10':{'alpha2':0, 'alpha4':0},
        'lssty1_mcal':{'alpha2':0, 'alpha4':0},
        'lssty10_mcal':{'alpha2':0, 'alpha4':0}
    }

    f, a = plt.subplots(
        7, 2,
        figsize=(3.35, 6.5),
        # sharex=True,
        gridspec_kw={'width_ratios':[1.75,1],'wspace':0.025,'hspace':0.1}
    )
    a[0,1].axis('off')

    lw = 1.75
    interpx = np.linspace(np.min(size_ratio), 2.1, 200)

    # ahist = a[0,0].twinx()
    ax, hist = plot_tratio(a[0,0], colors.g, colors.y, nbins=40)
    # ahist.set_ylim(top=6250)
    # ax.set_yticks([])

    for ax, key, imgkey, label in zip(
        a[1:],
        ['beta2', 'beta4', 'eta22', 'eta44', 'eta24', 'eta42'],
        range(7),
        [r'$\delta g / \delta e^{(2)}$',
         r'$\delta g / \delta e^{(4)}$',
         r'$\delta g / w_{22}$', r'$\delta g / w_{44}$',
         r'$\delta g / w_{24}$', r'$\delta g / w_{42}$']
        # [r'$\beta_2$', r'$\beta_4$', r'$\eta_{22}$', r'$\eta_{44}$', r'$\eta_{24}$', r'$\eta_{42}$']
    ):
        y_r, y_c = responses[key+'_r'], responses[key+'_c']
        y_r_mcal, y_c_mcal = mcal_responses[key+'_r'], mcal_responses[key+'_c']
        # need to integrate:
        interp_r = scipy.interpolate.interp1d(size_ratio, y_r, kind='cubic')
        interp_r_mcal = scipy.interpolate.interp1d(size_ratio, y_r_mcal, kind='cubic')

        for nepoch in [460, 46]:
            n, edges = hist[nepoch]
            bins = (edges[:-1] + edges[1:]) / 2
            n = n[bins>np.min(size_ratio)]
            bins = bins[bins>np.min(size_ratio)]
            coeff_r = scipy.integrate.trapezoid(y=interp_r(bins) * n,x=bins)
            coeff_r_mcal = scipy.integrate.trapezoid(y=interp_r_mcal(bins) * n,x=bins)
            # coeff_c = scipy.integrate.trapezoid(y=interp_c(bins) * n,x=bins)
            coeffs['lssty1' if nepoch==46 else 'lssty10'][key] = coeff_r
            coeffs['lssty1_mcal' if nepoch==46 else 'lssty10_mcal'][key] = coeff_r_mcal

        # ax[0].plot(
        #     interpx, interp_r(interpx),
        #     '--', lw=lw, color=colors.p,
        #     label=r'$\texttt{reGauss}$')
        ax[0].plot(
            interpx, interp_r_mcal(interpx),
            '-', lw=lw, color=colors.p,
            label=r'$\texttt{metacal}$')
        ax[0].set_ylabel(label)
        ax[1].imshow(
            imgs[imgkey],
            vmin=-np.max(imgs[imgkey]),
            vmax=np.max(imgs[imgkey]),
            cmap=colors.cmap_d)
        psflabel = [r'$\delta e^{(2)}_1$',
                    r'$\delta e^{(4)}_1$',
                    r'$\delta T^{(2)}$',
                    r'$\delta T^{(4)}$',
                    r'$\delta T^{(4)}$',
                    r'$\delta T^{(2)}$',
                    ][imgkey]
        psflabel_2 = ['','',
                      r'$e^{(2)}_1>0$',
                      r'$e^{(4)}_1>0$',
                      r'$e^{(2)}_1>0$',
                      r'$e^{(4)}_1>0$'][imgkey]
        ax[1].text(0.05, 0.8, psflabel, color='k', fontsize=9, transform=ax[1].transAxes)
        ax[1].text(0.05, 0.075, psflabel_2, color='k', fontsize=9, transform=ax[1].transAxes)

        # ax.plot(
        #     interpx, interp_c(interpx),
        #     ':', lw=lw, color=colors.p, label='complex')
        # if key in ['eta44', 'eta24', 'eta42']:
        #     ax.text(0.6, 0.82, label, fontsize=10, transform=ax.transAxes)
        # else:
        # ax[0].text(0.5, 0.82, label, ha='center', fontsize=10, transform=ax.transAxes)

    # a[1,0].legend(loc='lower right', borderaxespad=0.2)
    # a[0,0].yaxis.set_ticks_position('both')

    a[1,1].set_title(r'$\Delta$PSF')

    for ax in a[:-1,0]:
        ax.set_xticklabels([])

    for ax in a[1:,1]:
        ax.set_yticks([])
        ax.set_xticks([])

    # a[1,0].set_ylabel(r'$\delta c / \delta g^{(2)}$')
    # a[2,0].set_ylabel(r'$\delta c / w_{22}$')
    # a[3,0].set_ylabel(r'$\delta c / \delta e^{(4)}$')
    # a[4,0].set_ylabel(r'$\delta c / w_{44}$')
    # a[5,0].set_ylabel(r'$\delta c / w_{24}$')
    # a[6,0].set_ylabel(r'$\delta c / w_{42}$')
    # a[6,0].set_ylabel(r'$\delta c / \delta X_{\rm PSF}$')
    a[6,0].set_xlabel(r'$T_{\rm PSF}/T_{\rm gal}$')

    [ax.set_xlim(-.075, 2.175) for ax in a[:,0]]
    plt.subplots_adjust(right=0.99, bottom=0.075, top=0.975, left=0.2)
    plt.savefig('../figures/shear-moment-response-draft.jpg', dpi=300)
    plt.show()

    with open('../data/contamination-coeffs-empirical.json', 'w') as f:
        json.dump(coeffs, f, indent=4)
