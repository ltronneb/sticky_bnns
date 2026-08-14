import numpy as np
from scipy.stats import expon, bernoulli, t, mvn, norm
from scipy.optimize import minimize
import random
import math

# Function to compute the gradient of a standard Gaussian
def gradUx(x, Ux=None):
    """Compute the gradient of a standard Gaussian potential."""
    return x  # Gradient of 0.5 * sum(x**2)

# Rate on one dimension
def rateswitch(t, x0, v0):
    """Compute the switching rate at time t."""
    xt = x0 + v0 * t
    grad = gradUx(xt)
    lambda_vals = np.maximum(grad * v0, 0.0)
    return lambda_vals

# Global rate
def globalrate(t, x0, v0):
    """Compute the global switching rate at time t."""
    xt = x0 + v0 * t
    grad = gradUx(xt)
    lambda_vals = np.maximum(grad * v0, 0.0)
    return np.sum(lambda_vals)

# Optimization of the global rate
def myopt(upper_bound, x0val, v0val, eps=1e-9, round=True):
    """Optimize the global rate function."""
    def neg_globalrate(t):
        return -globalrate(t, x0=x0val, v0=v0val)
    
    if not round:
        result = minimize(neg_globalrate, 0, bounds=[(0, upper_bound)])
        Lambda_bar = -result.fun
        evals = result.nfev
    else:
        # First try with just 1 iteration
        result = minimize(neg_globalrate, 0, bounds=[(0, upper_bound)], options={'maxiter': 1})
        
        if result.success:
            Lambda_bar = -result.fun
            evals = result.nfev
        else:
            # Check the lower bound
            lambda_lower = globalrate(0, x0=x0val, v0=v0val)
            lambda_candidate = -result.fun
            
            if lambda_lower >= globalrate(eps, x0=x0val, v0=v0val) and lambda_lower > lambda_candidate:
                Lambda_bar = lambda_lower
                evals = result.nfev + 1
            else:
                # Check the upper bound
                lambda_upper = globalrate(upper_bound, x0=x0val, v0=v0val)
                
                if lambda_upper >= globalrate(upper_bound - eps, x0=x0val, v0=v0val) and lambda_upper > lambda_candidate:
                    Lambda_bar = lambda_upper
                    evals = result.nfev + 1
                else:
                    # Do a full optimization
                    result_long = minimize(neg_globalrate, 0, bounds=[(0, upper_bound)])
                    Lambda_bar = -result_long.fun
                    evals = result_long.nfev + 1
    
    output = {"Λbar": Lambda_bar, "evals": evals}
    return output

def zz(NS=None, x0_0=None, v0_0=None, tmax=None, B=False, roundopt=True, ε=1e-20):
    """Zigzag sampler algorithm."""
    # Check if the stopping criterion has been defined and how
    if NS is False and B is False:
        raise ValueError("Choose one stopping criterion: e.g. either: NS=10000; B=500000")
    elif NS is not False and B is not False:
        raise ValueError("No multiple stopping criteria: choose either NS=... or B=... set the other to false")
    
    # Get an estimate of the size of the skeleton (SS)
    SS = NS if NS is not False else B
    
    # Check the dimension of the problem
    if x0_0.ndim == 1:
        D = x0_0.shape[0]
    else:
        raise ValueError("The initial value x0_0 must be a vector, e.g. [x_01, x_02, ..., x_0D]")
    
    # Set up the arrays for the skeleton
    x0set = np.zeros((D, SS))
    v0set = np.zeros((D, SS))
    t0set = np.zeros((1, SS))
    GradEvals = np.zeros((3, SS))
    ErrorOpt = np.zeros((1, SS))
    
    # Setting up the state at the beginning of the location
    x0set[:, 0] = x0_0
    
    # Set velocity to one if missing
    if v0_0 is None:
        v0set[:, 0] = np.ones(D)
    else:
        v0set[:, 0] = v0_0
    
    t0set[0, 0] = 0.0
    
    x0i = x0set[:, 0].copy()
    v0i = v0set[:, 0].copy()
    ts = 0
    tp = 0
    horizon = tmax
    k = 1  # Python uses 0-indexing, so we start at 1 (2nd position)
    
    # Stopping criterion
    keepgoing = (k < NS) if NS is not False else np.sum(GradEvals) <= B
    
    while keepgoing:
        # Check for bounds at the horizon
        while np.isnan(globalrate(horizon, x0=x0i, v0=v0i)) and horizon > ε:
            horizon = horizon / 2
            # Count evaluations [added to the horizon change section]
            GradEvals[0, k] += 1
        
        # If approached the horizon (at the chosen ε₁) switch back
        if horizon <= ε:
            raise ValueError("Possible border ahead")
        else:  # Continue with the current horizon
            opt = myopt(upper_bound=horizon, x0val=x0i, v0val=v0i, round=roundopt)
            Λ_bar = opt["Λbar"]
            # Count evaluations [added to the optimization evaluation section]
            GradEvals[1, k] += opt["evals"]
            
            if Λ_bar == 0:  # Move deterministically if the rate of switching is 0
                ts += horizon
                x0i = x0i + horizon * v0i
            else:  # Propose a time for the switch
                tp = expon.rvs(scale=1/Λ_bar)
                
                if tp >= horizon:  # Move deterministically if reached the horizon
                    ts += horizon
                    x0i = x0i + horizon * v0i
                elif Λ_bar >= 1e10:
                    # Choose dimension to switch
                    lambda_vals = rateswitch(0, x0=x0i, v0=v0i)
                    m = np.argmax(lambda_vals)
                    
                    # Update location and switch velocity
                    x0i = x0i + tp * v0i
                    v0i[m] = -v0i[m]
                    
                    # Save the skeleton point
                    v0set[:, k] = v0i
                    x0set[:, k] = x0i
                    t0set[0, k] = t0set[0, k-1] + ts + tp
                    
                    # Reset time from skeleton point, horizon, and increase counter
                    ts = 0.0
                    tp = 0.0
                    horizon = tmax
                    k += 1
                    accept = True
                else:  # Evaluate proposal
                    accept = False
                    
                    while tp < horizon and not accept:
                        lambda_t = rateswitch(tp, x0=x0i, v0=v0i)
                        Lambda_t = np.sum(lambda_t)
                        ar = Lambda_t / Λ_bar
                        
                        # Count evaluations [added to the thinned section]
                        GradEvals[2, k] += 1
                        
                        if ar > 1:  # If optimization was wrong
                            horizon = tp
                            opt = myopt(upper_bound=horizon, x0val=x0i, v0val=v0i, round=roundopt)
                            Λ_bar = opt["Λbar"]
                            
                            # Restart with the new horizon/optimum
                            tp = expon.rvs(scale=1/Λ_bar)
                            
                            # Count evaluations [added to the optimization evaluation section]
                            GradEvals[1, k] += opt["evals"]
                            ErrorOpt[0, k] += 1
                        else:  # Evaluate acceptance
                            if bernoulli.rvs(ar):
                                # Choose dimension to switch based on rates
                                probs = lambda_t / Lambda_t
                                m = np.random.choice(D, p=probs)
                                
                                # Update location and switch velocity
                                x0i = x0i + tp * v0i
                                v0i[m] = -v0i[m]
                                
                                # Save the skeleton point
                                v0set[:, k] = v0i
                                x0set[:, k] = x0i
                                t0set[0, k] = t0set[0, k-1] + ts + tp
                                
                                # Reset time from skeleton point, horizon,
                                # flag acceptance and increase counter
                                ts = 0.0
                                tp = 0.0
                                horizon = tmax
                                k += 1
                                accept = True
                            else:  # Upon rejection increase stochastic time
                                tp += expon.rvs(scale=1/Λ_bar)
                    
                    if tp >= horizon and not accept:  # If exited while loop because horizon reached
                        ts += horizon
                        x0i = x0i + horizon * v0i
        
        # Update stopping criterion
        keepgoing = (k < NS) if NS is not False else np.sum(GradEvals) <= B
    
    # If we're using a budget, trim the unused entries
    if B is not False:
        x0set = x0set[:, :k]
        v0set = v0set[:, :k]
        t0set = t0set[:, :k]
        GradEvals = GradEvals[:, :k]
        ErrorOpt = ErrorOpt[:, :k]
    
    # Create combined output array for convenience
    outsk = np.vstack([t0set, x0set, v0set, GradEvals, ErrorOpt])
    
    # Create output dictionary
    output = {
        "SkeletonLocation": x0set,
        "SkeletonVelocity": v0set,
        "SkeletonTime": t0set,
        "GradientEvaluations": GradEvals,
        "ErrorsOptimization": ErrorOpt,
        "SK": outsk
    }
    
    return output

def zzsample(N, sk):
    """Sample points from the skeleton."""
    ts = sk["SkeletonTime"][0, :]
    vs = sk["SkeletonVelocity"]
    xs = sk["SkeletonLocation"]
    D = xs.shape[0]
    smpl = np.zeros((N, D))
    
    # Number of switching times (including 0)
    K = ts.shape[0]
    tm = (ts[-1] / N) * np.arange(1, N+1)  # Times of the sample
    
    for i in range(N):
        tm_i = tm[i]
        idx_i = np.where(ts <= tm_i)[0][-1]  # Last index where ts <= tm_i
        smpl[i, :] = xs[:, idx_i] + vs[:, idx_i] * (tm[i] - ts[idx_i])
    
    return smpl

def runHMC(epsilon, L, IT, qs, diagnose=False, Ux=None):
    """Run Hamiltonian Monte Carlo sampler."""
    dim = qs.shape[0]
    μmom = np.zeros(dim)
    Σmom = np.eye(dim)
    
    sample = np.zeros((IT+1, dim))
    accept = np.zeros((IT+1, 1))
    current_q = qs.copy()
    sample[0, :] = current_q
    
    if diagnose:
        qvals = np.zeros((L+1, dim, IT))
        pvals = np.zeros((L+1, dim, IT))
    
    for i in range(IT):
        # Set the initial position
        q = current_q.copy()
        p = np.random.multivariate_normal(μmom, Σmom)
        current_p = p.copy()
        
        if diagnose:
            # For visualization
            qplot = np.zeros((L+1, dim))
            pplot = np.zeros((L+1, dim))
            qplot[0, :] = q
            pplot[0, :] = p
        
        # Half step for the momentum
        p = p - epsilon * gradUx(q) / 2
        
        # Alternate full steps for positions and momentum
        for l in range(L-1):
            q = q + epsilon * p
            p = p - epsilon * gradUx(q)
            
            if diagnose:
                qplot[l+1, :] = q
                pplot[l+1, :] = p
        
        # For the L-th step make a full step for the position
        q = q + epsilon * p
        
        # And a half step for the momentum
        p = p - epsilon * gradUx(q) / 2
        
        if diagnose:
            qplot[L, :] = q
            pplot[L, :] = p
            qvals[:, :, i] = qplot
            pvals[:, :, i] = pplot
        
        # Negate the momentum (not really needed in practice)
        p = -p
        
        # Compute the acceptance rate
        current_U = 0.5 * np.sum(current_q**2)  # Using standard Gaussian
        current_K = np.sum(current_p**2) / 2
        proposed_U = 0.5 * np.sum(q**2)
        proposed_K = np.sum(p**2) / 2
        
        if np.random.uniform(0, 1) < np.exp(current_U - proposed_U + current_K - proposed_K):
            current_q = q.copy()
            accept[i+1, 0] = 1
        else:
            accept[i+1, 0] = 0
        
        sample[i+1, :] = current_q
    
    output = {"SampleQ": sample, "accept": accept}
    if diagnose:
        output["qvals"] = qvals
        output["pvals"] = pvals
        
    return output

def zzsummaries(dms, sk, B):
    """Calculate summaries for the zigzag sampler."""
    ts = sk["SkeletonTime"][0, :]
    vs = sk["SkeletonVelocity"][dms, :]
    xs = sk["SkeletonLocation"][dms, :]
    
    # Number of switching times (including 0)
    K = ts.shape[0]
    
    # First moment
    fms = np.zeros(K-1)
    for k in range(1, K):
        fms[k-1] = 0.5 * (ts[k] - ts[k-1]) * (vs[k-1] * (ts[k] - ts[k-1]) + 2 * xs[k-1])
    
    FM = (1/ts[-1]) * np.sum(fms)
    
    # Second moment
    sms = np.zeros(K-1)
    for k in range(1, K):
        sms[k-1] = ((vs[k-1] * (ts[k] - ts[k-1]) + xs[k-1])**3 - (xs[k-1]**3)) / (3 * vs[k-1])
    
    SM = (1/ts[-1]) * np.sum(sms)
    
    # Variance
    VAR = SM - FM**2
    
    # USE OF BATCH MEANS FOR SAMPLE VARIANCE
    FM_batches = np.zeros(B)
    
    for i in range(1, B+1):
        # Limits of the interval
        tau1_i = (ts[-1] / B) * (i - 1)
        tau2_i = (ts[-1] / B) * i
        
        # Index of the skeleton points contained in the interval
        idx_i = np.where((ts > tau1_i) & (ts <= tau2_i))[0]
        ni_i = len(idx_i) + 1
        
        # Lower limit of the integral
        a_i = np.zeros(ni_i)
        a_i[0] = tau1_i
        if len(idx_i) > 0:
            a_i[1:ni_i] = ts[idx_i]
        
        # Upper limit of the integral
        b_i = np.zeros(ni_i)
        if len(idx_i) > 0:
            b_i[:ni_i-1] = ts[idx_i]
        b_i[ni_i-1] = tau2_i
        
        # Velocities
        v_i = np.zeros(ni_i)
        idx_prev = np.where(ts <= tau1_i)[0]
        v_i[0] = vs[idx_prev[-1]] if len(idx_prev) > 0 else vs[0]
        if len(idx_i) > 0:
            v_i[1:ni_i] = vs[idx_i]
        
        # Locations
        x_i = np.zeros(ni_i)
        x_i[0] = xs[idx_prev[-1]] if len(idx_prev) > 0 else xs[0]
        if len(idx_i) > 0:
            x_i[1:ni_i] = xs[idx_i]
        
        # Initial skeleton point of the current interval
        t0_i = np.zeros(ni_i)
        t0_i[0] = ts[idx_prev[-1]] if len(idx_prev) > 0 else ts[0]
        if len(idx_i) > 0:
            t0_i[1:ni_i] = ts[idx_i]
        
        # Elements to be summed to obtain the first moment of the batch i
        fms_i = np.zeros(ni_i)
        for n in range(ni_i):
            fms_i[n] = -0.5 * ((a_i[n] - b_i[n]) * (v_i[n] * (a_i[n] + b_i[n] - 2 * t0_i[n]) + 2 * x_i[n]))
        
        FM_batches[i-1] = np.sqrt(B / ts[-1]) * np.sum(fms_i)
    
    # Sample variance calculation
    SGHAT = 1 / (B - 1) * np.sum((FM_batches - np.mean(FM_batches))**2)
    
    # Effective Sample Size
    ESS = ts[-1] * VAR / SGHAT
    
    output = {
        "FirstMoment": FM_batches, 
        "SecondMoment": SM, 
        "SampleVarianceBM": SGHAT, 
        "EffectiveSampleSize": ESS, 
        "Dimension": dms
    }
    
    return output

def ESSbm(smpl, nbatches):
    """Calculate Effective Sample Size using batch means."""
    N = smpl.shape[0]
    m = nbatches
    k = int(np.floor(N/m))
    
    # Batch means
    BM = np.zeros(m)
    for j in range(m):
        BM[j] = 1/k * np.sum(smpl[(j*k):((j+1)*k)])
    
    MU = 1/m * np.sum(BM)
    S2batch = 1/(m-1) * np.sum((BM - MU)**2)
    Varhatbatch = S2batch / m
    Varhatglobal = 1/(N-1) * np.sum((smpl - np.mean(smpl))**2)
    ESS = Varhatglobal / Varhatbatch
    
    return ESS

def ESStailprob(smpl, nbatches):
    """Calculate Effective Sample Size using tail probabilities."""
    N = smpl.shape[0]
    m = nbatches
    k = int(np.floor(N/m))
    
    # Tail probabilities
    smplneg = smpl.copy()
    smplneg[smpl > 0] = -smpl[smpl > 0]
    tailprob = t.cdf(smplneg, 1)
    
    # Batch means
    BM = np.zeros(m)
    for j in range(m):
        BM[j] = 1/k * np.sum(tailprob[(j*k):((j+1)*k)])
    
    MU = 1/m * np.sum(BM)
    S2batch = 1/(m-1) * np.sum((BM - MU)**2)
    Varhatbatch = S2batch / m
    Varhatglobal = 1/(N-1) * np.sum((tailprob - np.mean(tailprob))**2)
    ESS = Varhatglobal / Varhatbatch
    
    return ESS

# The functions below were part of the original code but require additional
# implementations that were not fully provided in the original snippet.
# I've included their signatures for completeness but they would need
# to be implemented based on the specific requirements.

def rateswitchj(t, j, x0, v0, grad=None, gradj=None, Uxj=None):
    """Compute rates for subsampling variant."""
    # Would need implementation of grad, gradj, Uxj
    pass

def getMestPOT(ss, x, v, tmax, J=None, D=None, SS_s=None):
    """Estimate maximum rate using Peaks-Over-Threshold method."""
    # Would need implementation of gpfit and other dependencies
    pass

def zz_w_ss(NS=None, x0_0=None, v0_0=None, tmax=None, B=False, ε=1e-20, 
            ssM=2000, NOBS=None, ssS=None):
    """Zigzag sampler with subsampling."""
    # Would need implementation of subsampling variant
    pass

# Example usage:
# x0 = np.array([0.0, 0.0, 0.0])  # Starting point
# v0 = np.array([1.0, 1.0, 1.0])  # Initial velocity
# result = zz(NS=1000, x0_0=x0, v0_0=v0, tmax=1.0)
# samples = zzsample(1000, result)
