"""
svdcut.py --- Correlations and SVD Cuts

This code illustrates the use of SVD cuts when calculating
correlations using random samples. See the Case Study in the
documentation for more information.
"""
from __future__ import print_function

import numpy as np
import gvar as gv

try:
    # may not be installed, in which case bail.
    import lsqfit
except:
    # fake the run so "make run" still works
    outfile = open('svdcut.out', 'r').read()
    print(outfile[:-1])
    exit()

SHOW_PLOTS = False

def main():
    gv.ranseed(4)
    x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    y_samples = [
        [2.8409,  4.8393,  6.8403,  8.8377, 10.8356, 12.8389, 14.8356, 16.8362, 18.8351, 20.8341],
        [2.8639,  4.8612,  6.8597,  8.8559, 10.8537, 12.8525, 14.8498, 16.8487, 18.8460, 20.8447],
        [3.1048,  5.1072,  7.1071,  9.1076, 11.1090, 13.1107, 15.1113, 17.1134, 19.1145, 21.1163],
        [3.0710,  5.0696,  7.0708,  9.0705, 11.0694, 13.0681, 15.0693, 17.0695, 19.0667, 21.0678],
        [3.0241,  5.0223,  7.0198,  9.0204, 11.0191, 13.0193, 15.0198, 17.0163, 19.0154, 21.0155],
        [2.9719,  4.9700,  6.9709,  8.9706, 10.9707, 12.9705, 14.9699, 16.9686, 18.9676, 20.9686],
        [3.0688,  5.0709,  7.0724,  9.0730, 11.0749, 13.0776, 15.0790, 17.0800, 19.0794, 21.0795],
        [3.1471,  5.1468,  7.1452,  9.1451, 11.1429, 13.1445, 15.1450, 17.1435, 19.1425, 21.1432],
        [3.0233,  5.0233,  7.0225,  9.0224, 11.0225, 13.0216, 15.0224, 17.0217, 19.0208, 21.0222],
        [2.8797,  4.8792,  6.8803,  8.8794, 10.8800, 12.8797, 14.8801, 16.8797, 18.8803, 20.8812],
        [3.0388,  5.0407,  7.0409,  9.0439, 11.0443, 13.0459, 15.0455, 17.0479, 19.0493, 21.0505],
        [3.1353,  5.1368,  7.1376,  9.1367, 11.1360, 13.1377, 15.1369, 17.1400, 19.1384, 21.1396],
        [3.0051,  5.0063,  7.0022,  9.0052, 11.0040, 13.0033, 15.0007, 16.9989, 18.9994, 20.9995],
        ]
    y = gv.dataset.avg_data(y_samples)
    svd = gv.dataset.svd_diagnosis(y_samples)
    y = gv.svd(y, svdcut=svd.svdcut)
    if SHOW_PLOTS:
        svd.plot_ratio(show=True)

    def fcn(p):
        return p['y0'] + p['s'] * x

    prior = gv.gvar(dict(y0='0(5)', s='0(5)'))
    fit = lsqfit.nonlinear_fit(data=y, fcn=fcn, prior=prior)
    print(fit)

if __name__ == '__main__':
    main()
