# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.special import gammainc

from tqdm import tqdm

def all_equal(iterable):
    """
    This function returns True if the elements of iterable are all equal.
    """
    it = iter(iterable)
    
    try:
        first = next(it)
    except StopIteration:
        return True
    
    return all(first == elem for elem in it)


def load_files(file_names, R=1, columns=None):
    """
    Reads data files and divides them in R replica, selecting the specified
    columns (observables).
    Returns a list of R * num_files data matrices, if num_files is the number of files given in input.
    """
    
    replica = [] # List of replica.
    
    # If columns is given and has only one element we need to reshape the data
    # matrix, in order to have the same shape in every case.
    reshape = len(columns) == 1 if columns is not None else False
    
    for name in tqdm(file_names, desc="Please wait while loading data"):
        print(name)
        file_tmp = np.loadtxt(name, usecols=columns, dtype=np.float64)

        # This is needed to guarantee that file_tmp is a two dimensional array.
        if reshape or len(file_tmp.shape) == 1:
            file_tmp = file_tmp.reshape((file_tmp.shape[0], 1))

        replica += np.array_split(file_tmp, R, axis=0)
        
    return replica



def add_mask(data, R, nalpha, nrep):
    """
    This function returns a mesked version of the matrix data according to replica sizes in Nrep.
    """
    mask = np.full((R, nalpha, max(nrep)), False, dtype='bool')

    for r in range(R):
        mask[r, nrep[r] : , :] = True

    return np.ma.array(data, mask=mask, dtype=np.float64)



def prepare_data(replica):
    """
    Returns:
        - Data matrix with shape (R,nrmax,nalpha) where
          R is the number of replica, nrmax the maximal size of the replica,
          nalpha is the number of primary observables;
        - A dictionary containing the following parameters: R, nalpha, nrmax,
          ntot (the total number of data), nrep (a list with the replica sizes);
        - Boolean value: True if replica sizes are all equal, False otherwise.

    """
    
    nrep   = [r.shape[0] for r in replica]
    nrmax  = max(nrep)
    ntot   = sum(nrep)
    nalpha = replica[0].shape[1]
    R      = len(nrep)
    

    data = np.zeros((R,nrmax,nalpha), dtype=np.float64)  # data matrix
    
    for i in range(R):
        data[i, : nrep[i], :] = replica[i]
    
    # Mask the data array in case replica sizes are different.
    # if not all_equal(nrep):
        # data = add_mask(data, R, nalpha, nrep)
        
    return (data, nalpha, nrep)

"""
Created on Sun Nov 28 21:26:14 2021

@author: s1833159
"""

class BaseSession(object):
    """
    This class handles a complete session of the program.
    Main methods:
        :func: `loadData`:      loads the input data and prepares them for
                                the analysis,
        :func: `loadDerived`:   loads the module and the methods of the derived
                                quantity,
        :func: `run`:           performs the analysis computing all the quantities
                                of interest,
        :func: `makePlots`:     makes plots and histograms.

    .. note::
        For further details on the methods, see the respective descriptions.
    """

    def __init__(self,
                 conf,
                 cache_file=None,
                 dry_run=False,
                 no_plots=False,
                 save_plots=False,
                 batch_mode=False,
                 backend=None):
        """
        
        Parameters
        ----------
        conf
            Contains the parameters passed by the user.
        cache_file
            If not None tells the class to write a cache file.
        dry_run
            If True the analysis is not perfomed.
        exc_info
            If True prints the traceback of the current execution.
        """
        
        self.dry_run    = dry_run
        self.conf       = conf
        self.cache_file = cache_file
        self.no_plots   = no_plots
        self.save_plots = save_plots
        self.batch_mode = batch_mode
        self.backend    = backend
        
        if self.batch_mode and self.backend is None:
            self.backend = 'Agg'

        
        self.setup()
        

    def setup(self):
        pass
    

    def loadData(self):
        """
        This method handles the loading of the input data.
        """
        
        data = load_files(self.conf['replica'],
                          self.conf['R'],
                          self.conf['primaries'])

        (data, nalpha, nrep) = prepare_data(data)
        
        self.data = data
        self.num_obs = nalpha
        self.rep_sizes = nrep
        self.num_rep = len(nrep)
            


class PrimarySession(BaseSession):
    def setup(self):
        if self.dry_run:
            return
        
        self.loadData()
        
        num_c = self.num_obs

        self.names = []
        
        if self.conf['primaries'] is not None:
            prim = [a+1 for a in self.conf['primaries']]
        else:
            prim = [a+1 for a in range(num_c)]
        
        for c in range(num_c):
            self.names.append("primary observable n. {}".format(prim[c]))
        
        self.analysis = PrimaryAnalysis(
            self.data, self.rep_sizes, name=self.names
        )
        
    def run(self):
        if self.dry_run:
            return
        
        self.analysis.mean()
        
        print("Running Gamma-method")
        
        self.analysis.errors(stau=self.conf['stau'])
    
        print("### ANALYSIS FINISHED ###")
        
        
        
#########################################

#-------------------------------------------------------------------------------
#       Compute gradient of derived quantity
#-------------------------------------------------------------------------------

def grad(f, abb, h):
    num_obs = abb.shape[0]
    fgrad = np.zeros(num_obs)
    ainc = abb.copy()

    for alpha in range(num_obs):
        if h[alpha] != 0:
            ainc[alpha]   = abb[alpha] + h[alpha]
            fgrad[alpha]  = f(ainc)
            ainc[alpha]   = abb[alpha] - h[alpha]
            fgrad[alpha] -= f(ainc)
            ainc[alpha]   = abb[alpha]
            fgrad[alpha] /= 2 * h[alpha]

    return fgrad



#-------------------------------------------------------------------------------
#       Error estimates of rho
#-------------------------------------------------------------------------------

def err_rho(N, t_max, w_opt, rho):
    """
    Takes:
        - N: total number of measurements.
        - t_max: parameter to truncate the series.
        - w_opt: optimal windowing value.
        - rho: normalised autocorrelation vector.

    Returns a vector with the error on the normalised autocorrelation.
    """

    ext_rho = np.zeros(2*t_max + w_opt + 1)
    err_rho = np.zeros(t_max + 1)
    ext_rho[:t_max + 1] = rho[:]

    for w in range(t_max + 1):
        for k in range(max(1, w - w_opt), w + w_opt + 1):
            err_rho[w] += (ext_rho[k+w] + ext_rho[abs(k-w)] - 2.0 * ext_rho[w] * ext_rho[k])**2
        err_rho[w] = math.sqrt(err_rho[w] / N)

    return err_rho



#-------------------------------------------------------------------------------
#       Error estimates of integrated autocorrelation time (tau)
#-------------------------------------------------------------------------------

def err_tau(N, t_max, tau_int_fbb):
    """
    Takes:
        - N: total number of measurements.
        - t_max: parameter to truncate the series.
        - tau_int_fbb: autocorrelation time.

    Returns the error of the autocorrelation time.
    """
    nf      = float(N)
    err_tau = np.zeros(t_max + 1)

    for w in range(t_max + 1):
        err_tau[w] = 2.0 * tau_int_fbb[w] * math.sqrt(w/nf)


    return err_tau



#-------------------------------------------------------------------------------
#       Bias cancellation for the derived quantity
#-------------------------------------------------------------------------------

def cancel_bias(fbb, fbr, fb, n_rep, sigma_f):
    """
    Takes:
        - fbb: mean values or derived value.
        - fbr: means computed on replica.
        - fb: weighed mean of replica means.
        - n_rep: list of replica sizes.
        - sigma_f: error of derived or primary.

    Returns fbb, fbr, fb without bias.

    For further details on this function see the article.
    """

    r = len(n_rep)
    n = sum(n_rep)

    if r >= 2:
        bf = (fb - fbb) / (r - 1)
        fbb -= bf
        if abs(bf) > sigma_f / 4:
            bias = bf / sigma_f
        fbr -= bf * n / n_rep
        fb -= bf * r

    return (fbb, fbr, fb)



#-------------------------------------------------------------------------------
#       Q value computation
#-------------------------------------------------------------------------------

def q_val(fbr, fb, n_rep, cfbb_opt):
    """
    Takes:
        - fbr: means computed on replica.
        - fb: weighed mean of replica means.
        - n_rep: list of replica sizes.
        - cfbb_opt: refined estimate of projected autocorrelation.

    Returns the Q-value of the replica distribution: goodness of fit to constant.
    """

    r = len(n_rep)

    if r >= 2:
        chisq = np.dot((fbr - fb)**2, n_rep) / cfbb_opt
        qval  = 1.0 - gammainc((r - 1) * 0.5, chisq * 0.5)
    else:
        qval = 0.0

    return qval



#-------------------------------------------------------------------------------
#       Implementation of the Gamma-method
#-------------------------------------------------------------------------------

def gamma(fbb, fbr, fb, delpro,
          r, n, n_rep,
          stau=1.5, rep_equal=False):
    """
    Takes:
        - obs: is a dictionary containing the mean values over all the data (fbb),
               over each replicum (fbr), over all the replica (fb) and
               the deviation from the mean value (delpro).
        - stau: guess for the ratio of tau/tauint. If 0, no autocorrelation is assumed.

    This function returns a dictionary with the results of the analysis:
        - W_opt: optimal windowing
        - t_max: maximum fixed time for the analysis
        - value: unbiased expectation value
        - valr: unbiased expectation value over each replicum
        - valb: unbiased expectation value over all replica,
        - dvalue: statiscal error of value,
        - ddvalue: error of dvalue,
        - tau_int: integrated autocorrelation time at W_opt,
        - dtau_int: error of tau_int,
        - tau_int_fbb: partial autocorrelation times.
        - dtau_int_fbb: error of tau_int_fbb.
        - rho: normalised autocorrelation function.
        - drho: error of rho.
        - qval: Q-value.
    """

    if stau == 0:  # No autocorrelations assumed.
        w_opt = 0
        t_max = 0
        flag  = False
    else:
        t_max = min(n_rep) // 2  # Do not take t larger than this.
                                 # // needed to take the floor of the division.
        flag  = True
        g_int = 0.0

    if rep_equal:
        weight = None
    else:
        weight = n_rep

    gamma_fbb = np.zeros(t_max + 2, dtype=np.float64)

    # values for W=0:
    gamma_fbb[1] = (delpro*delpro).mean()
    
    variance  = gamma_fbb[1] * n/(n-1)
    naive_err = np.sqrt(variance / n)

    t = 1

    while t <= t_max:
        for i in range(r):
            gamma_fbb[t + 1] += np.sum(delpro[i, 0 : n_rep[i] - t] * delpro[i, t : n_rep[i]])

        gamma_fbb[t + 1] /= (n - r * t)

        # Automatic windowing procedure

        if flag:
            g_int += gamma_fbb[t + 1] / gamma_fbb[1] # g_int(W) = tau_int(W) - 0.5

            if g_int <= 0.0: # No autocorrelation
                tauw = np.spacing(1.0) # Setting tau(W) to a tiny positive value
            else:
                tauw = stau / (np.log((g_int + 1) / g_int)) #è uguale a eq 20'beto

            gw = np.exp(-t / tauw) - tauw / np.sqrt(n * t)

            if gw < 0.0:  # g(W) has a minimum and this value of t is taken as the optimal value of W
                w_opt = t
                t_max = min(t_max, 2 * t)
                flag = False  # Gamma up to t_max

        t += 1
    # while-loop end
    print("t = %4d;\tGammaFbb = %.15e", t, gamma_fbb[t])

    # Here flag is True if windowing failed.
    if flag:
        w_opt = t_max

    gamma_fbb = gamma_fbb[1 : t_max + 2] #chi è gamma_fbb[0]: ora è il vecchio gamma_fbb[1]
    cfbb_opt  = gamma_fbb[0] + 2.0 * np.sum(gamma_fbb[1 : w_opt + 1])  # first estimate
    #eq 13 beto
    if cfbb_opt <= 0:
        raise print("Gamma pathological: estimated error^2 < 0")
        
    gamma_fbb   += cfbb_opt / n  # bias in Gamma corrected (eq: non numerata dopo eq:19 beto)
    cfbb_opt     = gamma_fbb[0] + 2 * np.sum(gamma_fbb[1:w_opt + 1])  # refined estimate
    #eq 13
    sigma_f      = np.sqrt(cfbb_opt / n)  # error of the expectation value of the observables
    #eq 14 beto
    rho          = gamma_fbb / gamma_fbb[0]  # normalized autocorrelation function
    print(n)
    drho         = err_rho(n, t_max, w_opt, rho)
    tau_int_fbb  = np.cumsum(rho) - 0.5
    dtau_int_fbb = err_tau(n, t_max, tau_int_fbb)
    
        
    # answers to be returned:
    
    (value, valr, valb) = cancel_bias(fbb, fbr, fb, n_rep, sigma_f)
    qval    = q_val(valr, valb, n_rep, cfbb_opt)
    dvalue  = sigma_f
    ddvalue = dvalue * np.sqrt((w_opt + 0.5) / n) # Statistical error of the error
    tau_int = tau_int_fbb[w_opt] # Equivalent to: cfbb_opt/2*gamma_fbb[0]
    dtau_int = tau_int * 2 * np.sqrt((w_opt - tau_int + 0.5) / n)

    return {       "w_opt": w_opt,
                   "t_max": t_max,
                   "value": value,
                  "dvalue": dvalue,
                 "ddvalue": ddvalue,
                 "tau_int": tau_int,
                "dtau_int": dtau_int,
             "tau_int_fbb": tau_int_fbb,
            "dtau_int_fbb": dtau_int_fbb,
                "variance": variance,
               "naive_err": naive_err,
                     "rho": rho,
                    "drho": drho,
                    "qval": qval,
                    "flag": flag}



class Formatter(object):
    def __call__(self, obj):
        return str(obj.__dict__)



class PrimaryFormatter(Formatter):
    def __call__(self, obj):
        stringified = ""

        for alpha in range(obj.num_obs):
            stringified += """\n\nResults for {name}:
         value: {value:.15e}
         error: {dvalue:.15e}
error of error: {ddvalue:.15e}
   naive error: {naive_err:.15e}
      variance: {variance:.15e}
       tau_int: {tau_int:.15e}
 tau_int error: {dtau_int:.15e}
         W_opt: {wopt:d}
         t_max: {tmax:d}
         Q_val: {qval:.15e}
            """.format(
                name=obj.name[alpha],
                value=obj.value[alpha],
                dvalue=obj.dvalue[alpha],
                variance=obj.variance[alpha],
                naive_err=obj.naive_err[alpha],
                ddvalue=obj.ddvalue[alpha],
                tau_int=obj.tau_int[alpha],
                dtau_int=obj.dtau_int[alpha],
                wopt=obj.w_opt[alpha],
                tmax=obj.t_max[alpha],
                qval=obj.qval[alpha]
            )

        return stringified



class DerivedFormatter(Formatter):
    def __call__(self, obj):
        return """\nResults for {name}:
         value: {value:.15e}
         error: {dvalue:.15e}
error of error: {ddvalue:.15e}
   naive error: {naive_err:.15e}
      variance: {variance:.15e}
       tau_int: {tau_int:.15e}
 tau_int error: {dtau_int:.15e}
         W_opt: {wopt:d}
         t_max: {tmax:d}
         Q_val: {qval:.15e}
        """.format(
            name=obj.name,
            value=obj.value,
            variance=obj.variance,
            naive_err=obj.naive_err,
            dvalue=obj.dvalue,
            ddvalue=obj.ddvalue,
            tau_int=obj.tau_int,
            dtau_int=obj.dtau_int,
            wopt=obj.w_opt,
            tmax=obj.t_max,
            qval=obj.qval
        )



class AnalysisData(object):
    def __init__(self, name=None, formatter=None):
        self.name = name
        self.formatter = formatter if formatter is not None else Formatter()

    def __str__(self):
        return self.formatter(self)



class Analysis(object):
    def __init__(self, data, rep_sizes, name=None, formatter=None):
        if data.ndim != 3:
            raise print("Data object should be a 3-dimensional array")

        self.name = name
        self.formatter = formatter
        self.data = data
        self.size = sum(rep_sizes)
        self.num_rep = data.shape[0]  # Number of replica.
        self.max_rep = data.shape[1]  # Size of the longest replicum.
        self.num_obs = data.shape[2]  # Number of primary observables.
        self.rep_sizes = rep_sizes
        self.rep_equal = all_equal(rep_sizes)
        self.weights = self.rep_sizes if not self.rep_equal else None

        self.results = AnalysisData(name=self.name, formatter=self.formatter)
        self.results.num_obs = self.num_obs

    def mean(self):
        if self.rep_equal:
            abr = np.mean(self.data, axis=1)
            abb = np.mean(abr, axis=0)
        else:
            abr = np.asarray(np.ma.mean(self.data, axis=1))
            abb = np.average(abr, axis=0, weights=self.weights)

        self.results.value = abb  # fbb
        self.results.rep_value = abr  # fbr
        self.results.rep_mean = np.average(
            abr, weights=self.weights, axis=0
        )  # fb
        self.results.deviation = self.data - abb  # delpro
        
        return self.results

    def errors(self, stau=1.5):
        print("Method 'errors' of '{}' "
                        "should implemented by subclasses".format(
                            self.__class__.__name__
                        ))
        return self.results



class PrimaryAnalysis(Analysis):
    def __init__(self, data, rep_sizes, name=None):
        super(PrimaryAnalysis, self).__init__(
            data,
            rep_sizes,
            name=name,
            formatter=PrimaryFormatter()
        )

    def errors(self, stau=1.5):
        r       = self.num_rep
        n       = self.size
        n_rep   = self.rep_sizes
        n_alpha = self.num_obs
        fbb     = self.results.value
        fbr     = self.results.rep_value
        fb      = self.results.rep_mean
        delpro  = self.results.deviation

        self.results.t_max     = [0]*n_alpha
        self.results.w_opt     = [0]*n_alpha
        self.results.variance  = [0]*n_alpha
        self.results.dvalue    = [0]*n_alpha
        self.results.ddvalue   = [0]*n_alpha
        self.results.naive_err = [0]*n_alpha
        self.results.tau_int   = [0]*n_alpha
        self.results.dtau_int  = [0]*n_alpha
        self.results.tau_int_fbb  = [0]*n_alpha
        self.results.dtau_int_fbb = [0]*n_alpha
        self.results.rho  = [0]*n_alpha
        self.results.drho = [0]*n_alpha
        self.results.qval = [0]*n_alpha
        
        for alpha in range(n_alpha):
            print("Computing errors for %s", self.name[alpha])

            res = gamma(fbb[alpha], fbr[:,alpha], fb[alpha], delpro[:,:,alpha],
                        r, n, n_rep,
                        stau, self.rep_equal)

            if res["flag"]:
                print("Windowing condition failed "
                                "for {} "
                                "up to W = {}".format(
                                    self.name[alpha],
                                    res["t_max"]
                                ))

            self.results.t_max[alpha]    = res["t_max"]
            self.results.w_opt[alpha]    = res["w_opt"]
            self.results.value[alpha]    = res["value"]
            self.results.variance[alpha]   = res["variance"]
            self.results.dvalue[alpha]   = res["dvalue"]
            self.results.ddvalue[alpha]  = res["ddvalue"]
            self.results.naive_err[alpha]  = res["naive_err"]
            self.results.tau_int[alpha]  = res["tau_int"]
            self.results.dtau_int[alpha] = res["dtau_int"]
            self.results.tau_int_fbb[alpha]  = res["tau_int_fbb"]
            self.results.dtau_int_fbb[alpha] = res["dtau_int_fbb"]
            self.results.rho[alpha]  = res["rho"]
            self.results.drho[alpha] = res["drho"]
            self.results.qval[alpha] = res["qval"]

        return self.results

    
def get_option_names(argv):
    options = dict()
    
    options["dry_run"]    = argv.get("--dry-run", False)
    options["no_plots"]   = argv.get("--no-plots", False)
    options["save_plots"] = argv.get("--save-plots", False)
    options["batch_mode"] = argv.get("--batch-mode", False)
    options["backend"]    = argv.get("--backend", None)
    options["conf_file"]  = argv.get("-f", None)
    options["cache_file"] = argv.get("--cache", None)
    options["directory"]  = argv.get("-d", None)
    options["replica"]    = argv.get("<file>", [])
    options["indices"]    = argv.get("--indices", None)
    options["ranges"]     = argv.get("--range", [])
    options["patterns"]   = argv.get("-p", [])
    options["params"]     = argv.get("-P", [])
    options["module"]     = argv.get("-m", None)
    options["functions"]  = argv.get("-q", [])
    options["primaries"]  = argv.get("-a", None)
    options["R"]          = argv.get("-R", 1)
    options["stau"]       = argv.get("-S", 1.5)

    return options
    
def datas(file):
    options = dict()
    options["R"] = 1
    options["replica"]    = [file]
    options["primaries"]  = None
    options["R"]          = 1
    options["stau"]       = 1.5
    print(options)
    
    test = PrimarySession(conf = options)
    test.run()
    
    return test.analysis.results