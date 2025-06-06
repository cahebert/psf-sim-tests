from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import numpy as np
from colors import color_scheme as colors

def plot_fpbin_param(ax, cat, camera, vmin, vmax, axcbar=None, numBins=200, cbarlabel='', cbartick=None):
    from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

    detectorIds = np.unique(cat["detector"])
    focalPlane_x = np.zeros(len(cat["x"]))
    focalPlane_y = np.zeros(len(cat["y"]))

    for detectorId in detectorIds:
        detector = camera[detectorId]
        map = detector.getTransform(PIXELS, FOCAL_PLANE).getMapping()

        detectorInd = cat["detector"] == detectorId
        points = np.array([cat["x"][detectorInd], cat["y"][detectorInd]])

        fp_x, fp_y = map.applyForward(points.astype(np.float64))
        focalPlane_x[detectorInd] = fp_x
        focalPlane_y[detectorInd] = fp_y

    binsx = np.linspace(focalPlane_x.min() - 1e-5, focalPlane_x.max() + 1e-5, numBins)
    binsy = np.linspace(focalPlane_y.min() - 1e-5, focalPlane_y.max() + 1e-5, numBins)

    statistic, x_edge, y_edge, bin_n = binned_statistic_2d(
        focalPlane_y,
        focalPlane_x,
        values=cat['z'],
        bins=[binsx, binsy]
        )
    binExtent = [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]]

    masked_statistic = np.ma.masked_where(statistic == 0, statistic)

    if (cat['z'] < 0).any():
        cmap = colors.cmap_d
    else:
        cmap = colors.cmap_s
    cmap.set_bad(color='white')

    sm = ax.imshow(
        masked_statistic,
        extent=binExtent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    if cbartick is None:
        ticks = None
    else:
        ticks = [-cbartick, 0, cbartick]
    if axcbar is not None:
        plt.colorbar(sm, cax=axcbar, label=cbarlabel, location='top', pad=0.05, ticks=ticks)
    else:
        plt.colorbar(sm, ax=ax, label=cbarlabel, location='top', pad=0.05, ticks=ticks)
    ax.set_aspect("equal")

    return ax

def plot_whisker(ax, cat, p, camera, scaling=1, keysize=.01, fontsize=8, keyposition=None):
    from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

    detectorIds = np.unique(cat["detector"])
    focalPlane_x = np.zeros(len(cat["x"]))
    focalPlane_y = np.zeros(len(cat["y"]))

    qdict = dict(alpha=1, angles='uv', pivot='middle', width=0.002,
                 headlength=0, headwidth=0, headaxislength=0, minlength=0)

    for detectorId in detectorIds:
        detector = camera[detectorId]
        map = detector.getTransform(PIXELS, FOCAL_PLANE).getMapping()

        detectorInd = cat["detector"] == detectorId
        points = np.array([cat["x"][detectorInd], cat["y"][detectorInd]])

        fp_x, fp_y = map.applyForward(points.astype(np.float64))
        focalPlane_x[detectorInd] = fp_x
        focalPlane_y[detectorInd] = fp_y

        bins = map.applyForward(np.vstack([np.linspace(0,4000,3),np.linspace(0,4000,3)]))

        h1, xedge, yedge, bin_n = binned_statistic_2d(
            fp_x, fp_y, bins=bins, values=cat[p+'1_pix'][detectorInd]
        )
        h2, xedge, yedge, bin_n = binned_statistic_2d(
            fp_x, fp_y, bins=bins, values=cat[p+'2_pix'][detectorInd]
        )

        e = np.hypot(h1, h2)
        beta = 0.5*np.arctan2(h2, h1)
        dx = e*np.cos(beta)
        dy = e*np.sin(beta)

        qdict.update({'color':'k'})
        q = ax.quiver((xedge[1:]+xedge[:-1])/2, (yedge[1:]+yedge[:-1])/2,
                        dx, dy, scale=scaling, **qdict)

    # if keysize<.01:
    #     if keysize * 1e3 >= 1:
    #         keylabel = f'{keysize*1e3:.0f}'+r'$\times 10^{-3}$'
    #     elif keysize * 1e4 >= 1:
    #         keylabel = f'{keysize*1e4:.0f}'+r'$\times 10^{-4}$'
    #     else:
    #         keylabel = f'{keysize*1e5:.2f}'+r'$\times 10^{-5}$'
    # else:
    keylabel = f'{keysize}'
    if keyposition is None:
        keyx, keyy = 240, 250
    else:
        keyx, keyy = keyposition
    ax.quiverkey(
        q, keyx, keyy, keysize,
        r'$|e|$ = '+keylabel,
        coordinates='data', labelpos='N', labelsep=0.05,
        fontproperties={'size':fontsize})

    ax.set_xlim(focalPlane_x.min(), focalPlane_x.max())
    ax.set_ylim(focalPlane_y.min(), focalPlane_y.max())

    ax.set_aspect("equal")
    return ax