# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:22:40 2025

@author: anton
"""

import numpy as np
import matplotlib.pyplot as plt

# parameters
mean = 0
std_dev=1

n_samples=1000


samples=np.random.normal(mean,std_dev,n_samples)

# plot data




#%% FIT GMM
from sklearn.mixture import GaussianMixture

gmm=GaussianMixture(n_components=1,reg_covar=1e-6)

gmm.fit(samples.reshape(-1,1))

# get distribution parameters from GMM

means=gmm.means_.flatten()[0]
stds=np.sqrt(gmm.covariances_).flatten()[0]

# Plotbereich
x = np.linspace(samples.min() - 1, samples.max() + 1, 500).reshape(-1, 1)

# GMM-Dichte (pdf) berechnen basierend auf 
logprob = gmm.score_samples(x)   # log(p(x))
pdf = np.exp(logprob)            # p(x)

sigma=1

plt.hist(samples,bins=10,color='black',density=True)
plt.plot(x, pdf, color='orange', linewidth=2)

plt.axvline(means,color='orange',linestyle='-')
plt.axvline(means - sigma * stds,color='orange',linestyle='--')
plt.axvline(means + sigma * stds,color='orange',linestyle='--')


plt.show()