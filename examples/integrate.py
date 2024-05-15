import gvar as gv
import numpy as np
import scipy.integrate

def trap_integral(f, interval, n=100):
    """ Estimate integral of f(x) on interval=(a,b) using the Trapezoidal Rule. """
    a, b = interval
    x = a + (b - a) * np.linspace(0, 1., n+1)
    fx = np.array([f(xi) for xi in x])
    I =  np.sum(fx[:-1] + fx[1:]) * (b - a) / (2. * n)
    return I

def integral(f, interval, tol=1e-8):
    """ GVar-compatible integrator from scipy routines """
    a, b = interval

    # collect GVar-valued parameters p and dI_dp for each p
    p = []
    dI_dp = []
    if isinstance(a, gv.GVar):
        p += [a]
        dI_dp += [-f(a).mean]
        a = a.mean
    if isinstance(b, gv.GVar):
        p += [b]
        dI_dp += [f(b).mean]
        b = b.mean

    # evaluate integral I of f(x).mean
    sum_fx = [0]
    def fmean(x):
        fx = f(x)
        if isinstance(fx, gv.GVar):
            sum_fx[0] += fx
            return fx.mean
        else:
            return fx
    I = scipy.integrate.quad(fmean, a, b, epsrel=tol)[0]

    # parameters from the integrand
    pf = gv.dependencies(sum_fx[0], all=True)

    # evaluate dI/dpf
    if len(pf) > 0:
        # vector-valued integrand returns df(x)/dpf
        def df_dpf(x):
            fx = f(x)
            if isinstance(fx, gv.GVar):
                return fx.deriv(pf)
            else:
                return np.array(len(pf) * [0.0])

        # integrate df/dpf to obtain dI/dpf
        dI_dpf = scipy.integrate.quad_vec(df_dpf, a, b, epsrel=tol)[0]

        # combine with other parameters, if any
        p += list(pf)
        dI_dp += list(dI_dpf)

    return gv.gvar_function(p, I, dI_dp) if len(p) > 0 else I

# integrands 
A = gv.gvar(2, 0.1)
K = gv.gvar(1, 0.11)
D = gv.gvar(1., 0.4)

def f(x):
    " easy "
    return A * np.cos(K * x**2 + D) ** 2

def g(x):
    " singular, not as easy"
    return A * x /(K * x**2 + 1e-6)

a = gv.gvar(0, 0.1)
b = gv.gvar(4, 0.1)

# Trapezoidal Rule
Itrap = trap_integral(f, (a, b), n=100)
print(f'Itrap = {Itrap:#P}')
Itrap_g = trap_integral(g, (a, b), n=100)
print(f'Itrap_g = {Itrap_g:#P}')
print()

# scipy routines
I = integral(f, (a, b))
print(f'I = {I:#P}')
I_g = integral(g, (a, b))
print(f'I_g = {I_g:#P}')
print()

# gvar.ode.integral
Iode = gv.ode.integral(f, (a, b))
print(f'Iode = {Iode:#P}')
Iode_g = gv.ode.integral(g, (a, b))
print(f'Iode_g = {Iode_g:#P}')
print()

# error budget
inputs = dict(a=a, b=b, A=A, K=K, D=D)
outputs = dict(I=I, Iode=Iode, Itrap=Itrap)
print(gv.fmt_errorbudget(inputs=inputs, outputs=outputs))
