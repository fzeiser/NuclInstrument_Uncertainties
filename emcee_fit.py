from __future__ import division
from math import pi
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import numpy as np
import emcee
import sys
import corner


# settings
Npar = 6 # number of parameters
datafile = "sigma_239Pu_gurevich_1976_g_abs.txt"

# comparable data
np.random.seed(123)

# constants

# import the data
# format: Ene,MeV     Sig,mb      dSig,mb
data = np.loadtxt(datafile)
x = data[:,0]
y = data[:,1]
yerr = data[:,2]

# fit function 
def SLO(E, E0, Gamma0, sigma0):
	# ad defined in Gurevich1976
	f =  sigma0 * E**2 * Gamma0**2 / ( (E**2 - E0**2)**2 + E**2 * Gamma0**2 )
	return f

def f(E, *pfit):
	E01, Gamma01, sigma01, E02, Gamma02, sigma02 = pfit
	f = SLO(E, E01, Gamma01, sigma01) + SLO(E, E02, Gamma02, sigma02)
	return f

# helper function to check whether a parameters is within some lowe/upper bounds
def in_bound(theta, bounds):
    ''' returns array with True/Fals, if theta is within the bounds'''
    theta_min = bounds[:,0]
    theta_max = bounds[:,1]
    theta_arr = np.array(theta)

    above_min = theta_min < theta_arr # True if theta is above the lower boundary
    below_max = theta_max > theta_arr # True if theta is below the higher boundary

    in_bound = np.logical_and(above_min,below_max)
    return in_bound

def analysis_emcee(x, y, yerr):
    # inital guess
               # omega, Gamma, sigma
               # MeV,   MeV,   mb
    p0 = (p0_bounds[:,0]+p0_bounds[:,1])/2.
    # p0=np.array([10.,2.,300.,  # 1st resonance
    #              12.,2.,300.]) # 2nd resonance
    # p0=np.array([11.,3.47,227.,  # 1st resonance
    #              14.,5.23,362.]) # 2nd resonance

    popt, pcov = curve_fit(f, xdata=x, ydata=y, sigma=yerr, p0=p0, bounds=(p0_bounds[:,0],p0_bounds[:,1]))
    # print popt, pcov
    perr = np.sqrt(np.diag(pcov))

    # for i in range(len(popt)):
    # 	print i, popt[i], perr[i]

    # Plot it
    # New Figure: Oslo type matrix
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
    ax.set_ylabel(r'$\sigma \, \mathrm{(mb)}$')

    Emin = 7  # MeV
    Emax = 20 # MeV
    E = np.linspace(Emin,Emax)

    # ax.errorbar(data[:,0],data[:,1],yerr=data[:,2],fmt="o", label="exp")
    # ax.errorbar(x,y,yerr=yerr,fmt="o", label="exp")
    # ax.plot(E,f(E,*p0), "--",label="initial guess")
    # ax.plot(E,f(E,*popt), label="fit result")

    # plt.show()
    plt.close(fig)

    # Try with emcee

    def chi2(theta,x, y, yerr):
    	model = f(x, *theta)
    	chi2 = np.sum( pow((y-model)/yerr,2) )
    	return chi2

    # Define the probability function lnprob as likelihood * prior.
    def lnprior(theta):
        # the current implementation corresponds to uniform priors within the boundaries
        theta_min = p0_bounds[:,0]
        theta_max = p0_bounds[:,1]
        theta_arr = np.array(theta)

        # check lower/higher boundary
        b_min = ((theta_min < theta_arr).sum() == theta_min.size)
        b_max = ((theta_max > theta_arr).sum() == theta_max.size)
        if not (b_min and b_max):
            return -np.inf
        return 0.0

    def lnlike(theta, x, y, yerr):
        return -0.5* chi2(theta, x, y, yerr)

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    # variables for emcee
    nWalkers = 20
    nSteps = 1000
    nBurnin = 500

    # Set up the sampler.
    # starting position for emcee: eg from minimization results
    p0_for_emcee = popt
    # dimensions of the variables to be estimated
    ndim = Npar     
    # for the gauss ball, see emcee desciption
    rand_factor = 1e-4 
    #initial position for each walker
    pos = [p0_for_emcee + rand_factor*np.random.randn(ndim) for i in range(nWalkers)]
    sampler = emcee.EnsembleSampler(nWalkers, ndim, lnprob, args=(x, y, yerr))

    # Clear and run the production chain.
    print("Running MCMC...")

    width = 60  # for progress bar
    progressBar = False # print a processBar
    for i, result in enumerate(sampler.sample(pos, iterations=nSteps, 
                                              rstate0=np.random.get_state()) ):
        if progressBar:
            #Progress Bar
            # add string "[NoLog!]" in the front so that we can exclude it from
            # being written into a log file (see eg. sample_folder/run_emcee.sh)
            n = int((width+1) * float(i) / nSteps)
            sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))

    # trow away a burn-in phase
    samples = sampler.chain[:, nBurnin:, :].reshape((-1, ndim))

    def timeplot(filename_basis=""):
        ndim = Npar
        Cols = 3
        Rows = ndim // Cols 
        Rows += ndim % Cols
        # Create a Position index
        Position = range(1,ndim + 1)

        fig = plt.figure(figsize=(20/2.,13/2.))
        for i in range(ndim):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(Rows,Cols,Position[i])
            # plt.plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
            plt.plot(sampler.chain[2:5, :, i].T, alpha=0.4) # plot only a few walkers
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.set_ylabel(parameter_names[i])
        # fig.tight_layout()
        # fig.savefig(filename_basis+"_time.png")
        # plt.close(fig)

    # timeplot()

    # Plot some samples onto the data.
    # Make x-axis array to plot from
    # fig = plt.figure()
    # ax = plt.subplot()
    # ax.errorbar(x,y,yerr=yerr,fmt="o", label="exp")

    # samples_rand_select = samples[np.random.randint(len(samples), size=200)]
    # for par in samples_rand_select:
    # 	ax.semilogy(E, f(E,*par), color="k", alpha=0.05)


    # Posterior distribuion / credibility intervals
    # (How often) are the "true" parameters within the credibility interval?


    # corner plots
    # fig = corner.corner(samples,labels=parameter_names, quantiles=[0.16, 0.5, 0.84],
    #                     show_titles=True, truths=p_true)
    # plt.show()
    # Compute the quantiles.
    quantiles = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    quantiles = np.array(quantiles)

    # Get the bounds from the quantiles
    bounds = np.empty((len(quantiles),2))
    bounds[:,0] = quantiles[:,0] - quantiles[:,2] # lower bound
    bounds[:,1] = quantiles[:,0] + quantiles[:,1] # upper bound

    # Histogram over how often the "true"-values is within the bounds
    # arr_inbounds = np.zeros(Npar)
    for i, is_in_bound in enumerate(in_bound(p_true, bounds)):
        if is_in_bound:
            arr_inbounds[i] += 1

def generate_data(x,p_true):
    N = len(x)
    y = f(x,*p_true)
    yerr_rel = relunc_min + relunc * np.random.rand(N)   # rel error is distributed unifor withing [relunc_min, relunc_min + relunc]
    yerr = y * yerr_rel
    y += yerr * np.random.randn(N)
    return x, y, yerr

# some dummy(!) true values
                    # omega, Gamma, sigma
                    # MeV,   MeV,   mb
p_true = np.array(  [10.1,   3.47,   227.,  # 1st resonance
                     13.,   5.23,    362.]) # 2nd resonance

# parameter names
parameter_names = [
"SLO1_E", "SLO1_gamma", "SLO1_sigma",
"SLO2_E", "SLO2_gamma", "SLO2_sigma"
]

#bounds for emcee
p0_bounds=[
     # omega, Gamma, sigma
     # MeV,   MeV,   mb
    (9.,14.), (1.,9.),  (150.,400.),     # (E)GDR number 1
    (12.,16.),(1,9.), (150,500),       # (E)GDR number 2
    ]
p0_bounds=np.array(p0_bounds)

if not np.all(in_bound(p_true,p0_bounds)):
    print in_bound(p_true,p0_bounds)
    raise ValueError("it's not your day: p_true is not within the bounds of emcee")


# some desctiption for the histogram
arr_inbounds = np.zeros(len(p_true))

E_arr = np.linspace(7,20,num=40) # xvalues for data --> choose from x-values of exp data
relunc_min = 0.05 # minimum relative uncertainty
relunc = 0.15 # relative uncertainty of the data points --> choose similar to exp data

# loop over data generation and analysis
nRuns = 100
for i in range(nRuns):
    print "run {0} of {1}".format(i,nRuns)
    x, y, yerr = generate_data(E_arr,p_true)

    # fig, ax = plt.subplots()
    # ax.set_yscale('log')
    # plt.errorbar(x,y,yerr=yerr,fmt="o", label="exp")
    # ax.plot(np.linspace(0,20),f(np.linspace(0,20),*p_true), label="input (wo uncertainties)")
    # plt.show()

    analysis_emcee(x, y, yerr)


arr_inbounds /= nRuns # fraction of times p_true was in the mcmc credibility interval

plt.figure()
plt.step(range(Npar),arr_inbounds,where="mid")

plt.xticks(np.arange(len(parameter_names)), parameter_names, rotation=70)
plt.tight_layout()

plt.show()