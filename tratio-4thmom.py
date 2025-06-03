import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from colors import color_scheme as colors
plt.style.use('./paper.mplstyle')

def plot_tratio(ax):
    import fitsio

    nbins = 41
    bins = np.linspace(0, 2, nbins)

    for nepoch, ls, c, ec, label in zip(
        [46, 460],
        ['-', '-'],
        [colors.p, colors.g],
        [colors.p, colors.g],
        ['Y1', 'Y10']
    ):
        data = fitsio.read(f'../data/summary-mcal-e{nepoch}-edges-wldb-varsize-gauss.fits')

        if nepoch == 46:
            ax.hist(
                data['T_ratio'],
                bins=bins,
                histtype='step',
                color='w',
                ls=ls,
                lw=0.25,
                # density=True,
                # label=label
                )
        ax.hist(
            data['T_ratio'],
            bins=bins,
            color=c,
            alpha=0.4 if nepoch==460 else 0.8,
            # density=True,
            label=label
            )

    ax.legend(borderaxespad=0.2)
    ax.set_ylabel(r'N$_{\rm gal}$')

    return ax

def plot_tratio_response(a):
    data = np.loadtxt('../data/tq_tratio_4thmom_012725.csv')  # load the TQ sims

    x = data[0][::-1]
    el = np.searchsorted(x, x[(x>0.6)&(x<2)][0])
    er = np.searchsorted(x, x[(x>0.6)&(x<2)][-1])

    xpts = np.linspace(0,3)

    for ax, coeff_tuple in zip(a, [(data[1][::-1], data[2][::-1]), (data[3][::-1], data[4][::-1])]):
        m = (coeff_tuple[0][er]-coeff_tuple[0][el]) / (x[er] - x[el])
        c = coeff_tuple[0][el+2] - m * (x[el+2] - x[0])
        ax.plot(xpts, xpts * m + c, color='lightgrey', lw=1.5)

        ax.plot(x, coeff_tuple[0], color=colors.g, ls='-', lw=2, label='real')
        ax.plot(x, coeff_tuple[1], color=colors.y, ls=':', lw=2, label='complex')

    a[0].legend(loc='lower left')

    a[0].set_ylabel(r'$\delta c / \delta e^{(4)}_{\rm PSF}$')
    a[1].set_ylabel(r'$\delta c / \left(e^{(4)} \frac{\delta T^{(4)} }{T^{(4)}}\right)_{\rm PSF}$')
    a[1].set_xlabel(r'$T_{\rm PSF} / T_{\rm gal}$')

    return a

plt.figure(figsize=(3.35, 4))
outer = gridspec.GridSpec(2, 1, height_ratios = [1, 2], hspace=0.1)
#make nested gridspecs
gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0])
gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[1], hspace = 0)

# f, a = plt.subplots(3, 1, figsize=(3.35, 4), sharex=True, gridspec_kw={'hspace':0})

a0 = plt.subplot(gs1[0])
plot_tratio(a0)

a1 = plt.subplot(gs2[0], sharex=a0)
a2 = plt.subplot(gs2[1], sharex=a0)
plot_tratio_response([a1, a2])

[plt.setp(ax.get_xticklabels(), visible=False) for ax in [a0, a1]]

a1.set_ylim(bottom=-0.1)
a2.set_ylim(bottom=-0.1)
a2.set_xlim([0,2.5])

plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)
# plt.savefig('../figures/tratio-4thmom-tq.jpg', dpi=300)
plt.show()