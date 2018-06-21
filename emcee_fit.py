from __future__ import division
from math import pi
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
import numpy as np
import emcee
import sys
import corner

# fit function 
def SLO(E, E0, Gamma0, sigma0):
	# ad defined in Gurevich1976
	f =  sigma0 * E**2 * Gamma0**2 / ( (E**2 - E0**2)**2 + E**2 * Gamma0**2 )
	return f

def f(E, *pfit):
	E01, Gamma01, sigma01, E02, Gamma02, sigma02, r_ = pfit # not that r is not used here; just a dummy!
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

    def chi2(theta,x, y, yerr):
        model = f(x, *theta)
        r = theta[-1]
        sigma2 = (yerr/r)**2 # note: if you remove r, this should be set to yerr**2
        chi2 = np.sum( pow((y-model),2)/sigma2 + np.log(sigma2)) # up to constants
        return chi2

    # Define the probability function lnprob as likelihood * prior.
    def lnprior(theta):
        # the current implementation corresponds to uniform priors within the boundaries
        if not np.all(in_bound(theta, p0_bounds)): # return inft if out of bounds
            return -np.inf
        return 0.0

    def lnlike(theta, x, y, yerr):
        return -0.5* chi2(theta, x, y, yerr)

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    # inital guess
               # omega, Gamma, sigma
               # MeV,   MeV,   mb
    p0 = (p0_bounds[:,0]+p0_bounds[:,1])/2.
    # p0=np.array([11.,3.47,227.,  # 1st resonance
    #              14.,5.23,362.]) # 2nd resonance

    # initial minimalization via optimum of the likelihood function 
    nll = lambda *args: -lnlike(*args) # maximize the likelihood = minimize negative (log) likelyhood
    result = minimize(nll, x0=p0, bounds=p0_bounds, args=(x, y, yerr),method="L-BFGS-B")
    popt = result["x"]

    # calculate variance from inverse hesse
    # see https://github.com/rickecon/StructEst_W17/blob/master/Notebooks/MLE/MLest.ipynb
    # section 4 in Maximum Likelihood Estimation (MACS 30100 and 40200)
    cov_maxll = result.hess_inv.todense()
    perr = np.sqrt(np.diag(cov_maxll))

    print "MaxLikelyhood result:"
    for i in range(len(popt)):
        print "{0} | {1:10s} \t {2:7.3f} +- {3:7.3f}".format(i, parameter_names[i], popt[i], perr[i])


    # Histogram over how often the "true"-values is within the bounds
    bounds = np.empty((Npar,2))
    bounds[:,0] = popt - perr # lower bound
    bounds[:,1] = popt + perr # upper bound
    for i, is_in_bound in enumerate(in_bound(p_true, bounds)):
        if is_in_bound:
            hist_inbound["minimize"][i] += 1

    def maxllplot():
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
        ax.set_ylabel(r'$\sigma \, \mathrm{(mb)}$')

        Emin = 7  # MeV
        Emax = 20 # MeV
        E = np.linspace(Emin,Emax)

        # ax.errorbar(data[:,0],data[:,1],yerr=data[:,2],fmt="o", label="exp")
        ax.errorbar(x,y,yerr=yerr,fmt="o", label="exp")
        # ax.plot(E,f(E,*p0), "--",label="initial guess")
        ax.plot(E,f(E,*popt), label="fit result")

    # maxllplot()
    # plt.show()
    
    # Try with emcee

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

    for i, result in enumerate(sampler.sample(pos, iterations=nSteps, 
                                              rstate0=np.random.get_state()) ):
        if progressBar:
            n = int((width+1) * float(i) / nSteps)
            sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))

    # throw away a burn-in phase
    samples = sampler.chain[:, nBurnin:, :].reshape((-1, ndim))

    # Some analysis plts
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

    def sampleMcmcPlot():
        # Plot some samples onto the data.
        fig = plt.figure()
        ax = plt.subplot()
        ax.errorbar(x,y,yerr=yerr,fmt="o", label="exp")

        samples_rand_select = samples[np.random.randint(len(samples), size=200)]
        for par in samples_rand_select:
        	ax.semilogy(E, f(E,*par), color="k", alpha=0.05)
        ax.plot(E,f(E,*p_true), "g--",label="true")

    # # Analysis plots of the mcmc runs
    # timeplot()
    # sampleMcmcplot()

    # Posterior distribuion / credibility intervals
    # (How often) are the "true" parameters within the credibility interval?

    def cornerplot():
        fig = corner.corner(samples,labels=parameter_names, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, truths=p_true) # for synthetic data
        # fig = corner.corner(samples,labels=parameter_names, quantiles=[0.16, 0.5, 0.84],
        #                 show_titles=True) # for exp data truth are not known
    # cornerplot()
    # plt.show()

    # Compute the quantiles.
    quantiles = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    quantiles = np.array(quantiles)

    # Histogram over how often the "true"-values is within the bounds
    bounds = np.empty((Npar,2))
    bounds[:,0] = quantiles[:,0] - quantiles[:,2] # lower bound
    bounds[:,1] = quantiles[:,0] + quantiles[:,1] # upper bound
    for i, is_in_bound in enumerate(in_bound(p_true, bounds)):
        if is_in_bound:
            hist_inbound["mcmc"][i] += 1

    print "\nEmcee result:"
    for i in range(Npar):
        print "{0} | {1:10s} \t {2:7.3f} + {3:7.3f} - {4:7.3f}".format(i, parameter_names[i], quantiles[i,0], quantiles[i,1], quantiles[i,2])

    # delete figure(s) -- they can jam the memmory 
    try:
        plt.close(fig)
    except NameError:
        pass

def generate_data(x,p_true):
    # generate synthetic data
    N = len(x)
    y = f(x,*p_true)
    yerr_rel = relunc_min + (relunc) * np.random.rand(N) # rel error is distributed unifor withing [relunc_min, relunc_min + relunc]
    yerr = y * yerr_rel
    y += yerr * np.random.randn(N)
    yerr *= p_true[-1] # under/overestimate reported error by scaling factor r
    return x, y, yerr

###################
###################

# comparable data
np.random.seed(123)

# settings
Npar = 7 # number of parameters (2*3SLO + 1 uncert.ratio)
datafile = "sigma_239Pu_gurevich_1976_g_abs.txt" # datafile if "real data" is used

# import the data
# format: Ene,MeV     Sig,mb      dSig,mb
data = np.loadtxt(datafile)

x = data[:,0]
y = data[:,1]
yerr = data[:,2]

# some emcee settings
width = 60  # for progress bar
progressBar = False # print a processBar
nWalkers = 20
nSteps =  4000
nBurnin = 2000

# parameter names
parameter_names = [
"SLO1_E", "SLO1_gamma", "SLO1_sigma",   # 1st resonance
"SLO2_E", "SLO2_gamma", "SLO2_sigma",   # 2nd resonance
"r"                                 # ln(r) (ln of fraction of underestimation)
]

# bounds for emcee & minimize
p0_bounds=np.array([
     # omega, Gamma, sigma
     # MeV,   MeV,   mb
    (9.,12.), (1.,9.),  (150.,400.),     # (E)GDR number 1
    (12.,16.),(1,9.), (150,500),         # (E)GDR number 2
    (0.5,1.5)                            # r (std.deviation scaling)
    ])
 
# for synthetic data: true values
                    # omega, Gamma, sigma
                    # MeV,   MeV,   mb
p_true = np.array(  [10.1,   3.47,   227.,  # 1st resonance
                     13.,   5.23,    362.,  # 2nd resonance
                     1.2])                  # r (std.deviation scaling)

# parameters for uncertainty to be created
# more realistic: choose from x-values of exp data
relunc_min = 0.1 # minimum relative uncertainty
relunc = 0.2    # relative uncertainty of the data points,

# if using synthetic data: Make sure that true values are within the bounds
if not np.all(in_bound(p_true,p0_bounds)):
    print in_bound(p_true,p0_bounds)
    raise ValueError("it's not your day: p_true is not within the bounds of emcee")

# for plotting and synthetic data
E_arr = np.linspace(7,20,num=60) # xvalues for data --> more realistic: choose from x-values of exp data

# if using synthetic data: array/histogram which is filled every time 
# p_true is within the credibility interval
hist_inbound = dict(mcmc=np.zeros(Npar), minimize=np.zeros(Npar))

# loop over (data generation) and analysis
nRuns = 50 # can be set to 1 for analysis of exp data
for i in range(nRuns):
    print "\nrun {0} of {1}\n".format(i,nRuns)

    # comment out if using exp data
    x, y, yerr = generate_data(E_arr,p_true)

    # plot generated data
    # fig, ax = plt.subplots()
    # ax.set_yscale('log')
    # plt.errorbar(x,y,yerr=yerr,fmt="o", label="exp")
    # ax.plot(np.linspace(0,20),f(np.linspace(0,20),*p_true), label="input (wo uncertainties)")
    # plt.show()

    analysis_emcee(x, y, yerr)

# Plot histogram of how often p_true is within confidence/credibility interval
for key, hist in hist_inbound.iteritems(): 
    hist /= nRuns # fraction of times p_true was in the mcmc credibility interval
    plt.figure()
    plt.title("{0}: fraction of sucesses".format(key))
    plt.step(range(Npar),hist,where="mid")
    plt.xticks(np.arange(len(parameter_names)), parameter_names, rotation=70)
    plt.tight_layout()

plt.show()