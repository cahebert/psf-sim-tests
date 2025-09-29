from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import numpy as np
from colors import color_scheme as colors

def pix_to_fp(points, det):
    from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
    map = det.getTransform(PIXELS, FOCAL_PLANE).getMapping()
    fpx, fpy = map.applyForward(points.astype(np.float64))
    return fpx, fpy

def plot_fpbin_param(ax, cat, camera, vmin, vmax, axcbar=None, numBins=10, cbarlabel='', cbartick=None, cmap=None):
    binned_result = get_fullfp_bins(cat, camera, numBins, columns=['z'])

    if cmap is None:
        if (cat['z'] < 0).sum() > 50:
            cmap = colors.cmap_d
        else:
            cmap = colors.cmap_s

    if vmin is None or vmax is None:
        vals = [res['stat'] for d, res in binned_result['z'].items()]
        vmin = np.nanquantile(vals, 0.1)
        vmax = np.nanquantile(vals, 0.99)

    for detId, res in binned_result['z'].items():
        x, y, z = res['x'], res['y'], res['stat']
        sm = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap=cmap)

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

def get_fullfp_bins(cat, camera, nBins, columns=[]):
    assert len(columns) > 0, "Must provide at least one column to bin by."
    detectorIds = cat["detector"].unique()

    result = {col:{det:{} for det in detectorIds} for col in columns}
    for detectorId in detectorIds:
        detector = camera[detectorId]
        detectorInd = cat["detector"] == detectorId

        points = np.array([cat["x"][detectorInd], cat["y"][detectorInd]])
        fp_x, fp_y = pix_to_fp(points, detector)

        for col in columns:
            bins_x = np.linspace(fp_x.min() - 1e-5, fp_x.max() + 1e-5, nBins+1)
            bins_y = np.linspace(fp_y.min() - 1e-5, fp_y.max() + 1e-5, nBins+1)
            statistic, x_edge, y_edge, _ = binned_statistic_2d(
                fp_x, fp_y,
                values=cat[col][detectorInd],
                bins=[bins_x, bins_y]
                )
            s2n, _, _, _ = binned_statistic_2d(
                fp_x, fp_y,
                values=cat['am_s2n'][detectorInd],
                bins=[bins_x, bins_y],
                )
            result[col][detectorId]['stat'] = statistic.T
            result[col][detectorId]['s2n'] = s2n.T
            result[col][detectorId]['x'] = x_edge
            result[col][detectorId]['y'] = y_edge

    return result


def get_whisker(e1, e2):
    """Get dx,dy / u,v for a whisker plot."""
    e = np.hypot(e1, e2)
    beta = 0.5 * np.arctan2(e2, e1)
    dx = e * np.cos(beta)
    dy = e * np.sin(beta)
    return dx, dy

# def get_fullfp_bins(cat, camera, nBins, columns=[], adaptiveBinning=True):
#     assert len(columns) > 0, "Must provide at least one column to bin by."

#     from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

#     detectorIds = cat["detector"].unique()
#     focalPlane_x = np.zeros(len(cat["x"]))
#     focalPlane_y = np.zeros(len(cat["y"]))

#     for detectorId in detectorIds:
#         detector = camera[detectorId]
#         map = detector.getTransform(PIXELS, FOCAL_PLANE).getMapping()

#         detectorInd = cat["detector"] == detectorId
#         points = np.array([cat["x"][detectorInd], cat["y"][detectorInd]])

#         fp_x, fp_y = map.applyForward(points.astype(np.float64))
#         focalPlane_x[detectorInd] = fp_x
#         focalPlane_y[detectorInd] = fp_y

#     if adaptiveBinning:
#         # Use a course 32x32 binning to determine the mean source density
#         # in regions where there are sources.
#         binsx = np.linspace(focalPlane_x.min() - 1e-5, focalPlane_x.max() + 1e-5, 33)
#         binsy = np.linspace(focalPlane_y.min() - 1e-5, focalPlane_y.max() + 1e-5, 33)

#         binnedNumSrc = np.histogram2d(focalPlane_x, focalPlane_y, bins=[binsx, binsy])[0]
#         meanSrcDensity = np.mean(binnedNumSrc, where=binnedNumSrc > 0.0)

#         numBins = int(np.round(16.0 * np.sqrt(meanSrcDensity)))
#         numBins = max(numBins, nBins)
#     else:
#         numBins = nBins

#     binsx = np.linspace(focalPlane_x.min() - 1e-5, focalPlane_x.max() + 1e-5, numBins+1)
#     binsy = np.linspace(focalPlane_y.min() - 1e-5, focalPlane_y.max() + 1e-5, numBins+1)

#     result = {}
#     for col in columns:
#         statistic, x_edge, y_edge, _ = binned_statistic_2d(
#             focalPlane_x,
#             focalPlane_y,
#             values=cat[col],
#             bins=[binsx, binsy]
#             )
#         result[col] = statistic.T
#     return result, x_edge, y_edge

def plot_whisker(ax, cat, p, camera, axcbar=None,
                 scaling=1, keysize=.01, fontsize=8, keyposition=None):
    qdict = dict(alpha=1, angles='uv', pivot='middle', width=0.002,
                 headlength=0, headwidth=0, headaxislength=0, minlength=0)

    binned_result = get_fullfp_bins(cat, camera, nBins=3, columns=[p+'_1_pix', p+'_2_pix'])
    binned_e1 = binned_result[p+'_1_pix']
    binned_e2 = binned_result[p+'_2_pix']

    s2n_all = [binned_e1[d]['s2n'] for d in binned_e1.keys()]
    vmin = np.nanquantile(s2n_all,0)
    vmax = np.nanquantile(s2n_all,.95)
    # print('s2n range: ', vmin, vmax)
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = colors.cmap_s
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)

    for detId, bin_e1 in binned_e1.items():
        bin_e2 = binned_e2[detId]

        dx, dy = get_whisker(bin_e1['stat'], bin_e2['stat'])
        qdict.update({'cmap':colors.cmap_s, 'norm':norm})

        # qdict.update({'color':'k'})
        q = ax.quiver(
            (bin_e2['x'][1:]+bin_e2['x'][:-1])/2,
            (bin_e2['y'][1:]+bin_e2['y'][:-1])/2,
            dx, dy,
            bin_e1['s2n'],
            scale=scaling, **qdict
        )

    if keysize < 1e-3:
        if keysize == 1e-4:
            keylabel = r'$10^{-4}$'
        elif keysize == 1e-5:
            keylabel = r'$10^{-5}$'
        else:
            keylabel = f'{keysize}'
    else:
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

    xs = [(np.max(b['x']), np.min(b['x'])) for d, b in binned_e2.items()]
    ys = [(np.max(b['y']), np.min(b['y'])) for d, b in binned_e2.items()]
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))

    if axcbar is not None:
        plt.colorbar(sm, cax=axcbar, label='S/N', location='bottom', pad=0.05)

    ax.set_aspect("equal")
    return ax