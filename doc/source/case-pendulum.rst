.. |GVar| replace:: :class:`gvar.GVar`

.. |~| unicode:: U+00A0
   :trim:


Case Study:  Numerical Analysis --- Pendulum Clock
=====================================================
This case study illustrates how to mix |GVar|\s with numerical routines
for integrating differential equations (:any:`gvar.ode <ode>`) and for
finding roots of functions (:any:`gvar.root <root>`). It also gives a
simple example of a simulation that uses |GVar|\s.

The Problem
----------------

The precision of a particular pendulum clock is limited by two  dominant
factors: 1) the length of the pendulum (0.25m) can be adjusted  with a precision
of at best ±0.5mm; and 2) irregularities in  the drive mechanism mean that the
maximum angle of swing (π/6) is uncertain by ±0.025 |~| radians. The challenge
is to determine how these uncertainties affect time-keeping over a day.

The angle ``theta(t)`` of the pendulum satisfies a differential equation ::

    d/dt d/dt theta(t) = -(g/l) sin(theta(t))

where ``g`` is the acceleration due to gravity and the ``l`` is the length
of the pendulum.


Pendulum Dynamics; Finding the Period
---------------------------------------
We start by designing code to integrate the differential equation::

    import numpy as np
    import gvar as gv

    def make_pendulum(theta0, l):
        """ Create pendulum solution y(t) = [theta(t), d/dt theta(t)].

        Initial conditions are y(0) = [theta0, 0]. Parameter l is the
        length of the pendulum.
        """
        g_l = 9.8 / l
        def deriv(t, y):
            """ Calculate d/dt [theta(t), d/dt theta(t)]. """
            theta, dtheta_dt = y
            return np.array([dtheta_dt, - g_l * gv.sin(theta)])
        y0 = np.array([theta0, 0.0])
        return gv.ode.Integrator(deriv=deriv).solution(0.0, y0)

Given a solution ``y(t)`` of the differential equation from this method,
we find the period of oscillation using :mod:`gvar.root`: the period
is the time at which the pendulum returns to its starting point and its
velocity (``y(t)[1]``) vanishes::

    def find_period(y, Tapprox):
        """ Find oscillation period of pendulum solution y(t).

        Parameter Tapprox is the approximate period. The code finds the time
        between 0.7 * Tapprox and 1.3 * Tapprox where y(t)[1] = d/dt theta(t)
        vanishes. This is the period, provided Tapprox is correctly chosen.
        """
        def dtheta_dt(t):
            """ vanishes when dtheta/dt = 0 """
            return y(t)[1]
        return  gv.root.refine(dtheta_dt, (0.7 * Tapprox, 1.3 * Tapprox))

Analysis
-----------
The last piece of the code does the analysis::

    def main():
        l = gv.gvar(0.25, 0.0005)               # length of pendulum
        theta_max = gv.gvar(np.pi / 6, 0.025)   # max angle of swing
        y = make_pendulum(theta_max, l)         # y(t) = [theta(t), d/dt  theta(t)]

        # period in sec
        T = find_period(y, Tapprox=1.0)
        print('period T = {} sec'.format(T))

        # uncertainty in minutes per day
        fmt = 'uncertainty = {:.2f} min/day\n'
        print(fmt.format((T.sdev / T.mean) * 60. * 24.))

        # error budget for T
        inputs = dict(l=l, theta_max=theta_max)
        outputs = dict(T=T)
        print(gv.fmt_errorbudget(outputs=outputs, inputs=inputs))

    if __name__ == '__main__':
        main()

Here both the length of the pendulum and the maximum angle of swing
have uncertainties and are represented by |GVar| objects. These uncertainties
work their way through both the integration and root finding to give
a final result for the period that is also a |GVar|. Running the code
results in the following output::

    period T = 1.0210(20) sec
    uncertainty = 2.79 min/day

    Partial % Errors:
                       T
    --------------------
            l:      0.10
    theta_max:      0.17
    --------------------
        total:      0.19

The period is ``T = 1.0210(20) sec``, which has an uncertainty of
about ±0.2%. This corresponds to an uncertainty of ±2.8 |~| min/day
for the clock.

The uncertainty in the period is caused by the uncertainties in the
length |~| ``l`` and the angle of maximum swing |~| ``theta_max``.
The error budget at the end of the output shows how much error comes
from each source: 0.17% comes from the angle, and 0.10% comes from
the length. (The two errors added in quadrature give the total.)
We could have estimated the error due to the length from the
standard formula 2π  |~| sqrt(*l*/*g*) for the period, which is approximately
true here. Estimating the uncertainty due to the angle is trickier, since it
comes from nonlinearities in the differential equation.

The error budget tells us how to improve the clock. For example, we can
reduce the error due to the angle by redesigning the clock so that the
maximum angle of swing is π/36 |~| ± |~| 0.025 rather
than |~| π/6 |~| ± |~| 0.025.
The period becomes independent of the maximum angle as that angle vanishes,
and so becomes less sensitive to uncertainties in it. Taking the smaller angle
reduces that part of the period's error from 0.17% to 0.03%, thereby cutting
the total error almost in half, to ±0.10% or about ±1.5 |~| min/day. Further
improvement requires tighter control over the length of the pendulum.

Simulation
------------
We can check the error propagation analysis above using
a simulation. Adding the following code at the end of ``main()`` above ::

    # check errors in T using a simulation
    Tlist = []
    for i in range(100):
        y = make_pendulum(theta_max(), l())
        T = find_period(y, Tapprox=1.0)
        Tlist.append(T)
    print('period T = {:.4f} +- {:.4f}'.format(np.mean(Tlist), np.std(Tlist)))

gives the following additional output::

    period T = 1.0209 +- 0.0020

The new code generates 100 different values for the period ``T``, corresponding
to randomly chosen values for ``theta_max`` and ``l`` drawn from the
Gaussian distributions corresponding to their |GVar|\s. (In general, each
call ``x()`` for |GVar| ``x`` is a new random number drawn from ``x``'s
Gaussian distribution.)
The mean and
standard deviation of the list of periods give us our final result.
Results fluctuate with only 100 samples; taking 10,000 samples shows that
the result is 1.0210(20), as we obtained
in the previous section above (using a tiny fraction of the computer time).

Note that the |GVar|\s in this simulation are uncorrelated and so their random
values can be generated independently. :func:`gvar.raniter` should be used  to
generate random values from correlated |GVar|\s.
