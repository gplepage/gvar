"""
pendulum.py --- GVars with numerical ode integration and root finding.

This code calculates the uncertainty in the period of a pendulum  given
uncertainties in its length and in the angle of release.  It uses
gvar.ode.Integrator to integrate the pendulum's equation  of motion

    d/dt d/dt theta(t) = - (g/l) sin(theta(t)),

and gvar.root.refine  to find the period. The uncertainties in l and
theta(0) are propagated through the integration and root-finding
algorithms by gvar. The impact on the time-keeping abilities of the
clock housing this pendulum are also examined.
"""

from __future__ import print_function   # makes this work for python2 and 3

import gvar as gv
import numpy as np

gv.ranseed((1,2,4)) # gives reproducible random numbers

def main():
    l = gv.gvar(0.25, 0.0005)               # length of pendulum
    theta_max = gv.gvar(np.pi / 6, 0.025)   # max angle of swing
    y = make_pendulum(theta_max, l)         # y(t) = [theta(t), d/dt  theta(t)]
    T = find_period(y, Tapprox=1.0)
    print('period T = {} sec'.format(T))
    fmt = 'uncertainty = {:.2f} min/day\n'
    print(fmt.format((T.sdev / T.mean) * 60. * 24.))

    # error budget for T
    inputs = gv.BufferDict() # dict(l=l, theta_max=theta_max)
    inputs['l'] = l
    inputs['theta_max'] = theta_max
    outputs = {'T':T}
    print(gv.fmt_errorbudget(outputs=outputs, inputs=inputs))

    # check errors in T using a simulation
    Tlist = []
    for i in range(200):
        y = make_pendulum(theta_max(), l())
        Tlist.append(find_period(y, Tapprox=1.0))
    print('period T = {:.4f} +- {:.4f}'.format(np.mean(Tlist), np.std(Tlist)))

def find_period(y, Tapprox):
    """ Find oscillation period of y(t).

    Parameter Tapprox is the approximate period. The code finds the time
    between 0.7 * Tapprox and 1.3 * Tapprox where y(t)[1] = d/dt theta(t)
    vanishes. This is the period.
    """
    def dtheta_dt(t):
        """ vanishes when dtheta/dt = 0 """
        return y(t)[1]
    return  gv.root.refine(dtheta_dt, (0.7 * Tapprox, 1.3 * Tapprox))

def make_pendulum(theta_max, l):
    """ Create ode solution y(t) = [theta(t), d/dt theta(t)].

    Initial conditions are y(0) = [theta_max, 0]. Parameter l is the
    length of the pendulum.
    """
    g_l = 9.8 / l
    def deriv(t, y):
        """ Calculate [d/d theta(t), d/dt d/dt theta(t)]. """
        theta, dtheta_dt = y
        return np.array([dtheta_dt, - g_l * gv.sin(theta)])
    y0 = np.array([theta_max, 0.0])
    return gv.ode.Integrator(deriv=deriv).solution(0.0, y0)

if __name__ == '__main__':
    main()

# Created by G. Peter Lepage (Cornell University) on 2014-04-28.
# Copyright (c) 2015 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

