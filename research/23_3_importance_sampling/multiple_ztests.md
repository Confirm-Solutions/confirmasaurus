We explore importance sampling on some trivial test cases.

The steps of importance sampling are described in the overleaf, Appendix D.

We will implement them first for the 1-dimensional z-test, then the 2- and 3-dimensional z-tests.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas 
import seaborn 
import sklearn 
#import ipython
from sklearn.cluster import KMeans
```

```python
mu = np.linspace(-2, 2, 11)
z = np.random.normal(0, 1, 1000)
data = mu[None,:] + z[:,None]
```

```python
flat_data =  data.flatten()
#Now, we select the rejections in order to run k-means on them
selection = flat_data > 1.96
rejections = flat_data[selection]
standardized_rejections = (rejections - np.mean(rejections))/np.std(rejections)
```

```python
n_clusters = 5
kmeans= KMeans(n_clusters=n_clusters,
               init = "random",
            n_init=10,
               max_iter=300,
               random_state = 42)
```

```python
kmeans.fit(standardized_rejections.reshape(-1,1))
```

```python
mu_cluster_centers = kmeans.cluster_centers_ * np.std(rejections) + np.mean(rejections)
mu_cluster_centers
```

```python
kmeans.labels_
```

```python
n_orig = len(flat_data)
flat_labels = np.full(n_orig, -1,dtype = np.int32)
flat_labels[selection] = kmeans.labels_
np.unique(flat_labels, return_counts=True)
```

```python
labels = flat_labels.reshape(data.shape)
n_sims_per_theta = data.shape[0] # 1000 for now
n_theta = data.shape[1] # 11 for now
target_fraction = np.full((n_clusters + 1, n_theta),-1)
labelset = np.unique(labels)
labelbins = np.append(labelset - 0.5,n_clusters - 0.5)
for i in range(n_theta):
    target_fraction[:,i] = np.histogram(labels[:,i], bins = labelbins)[0]
```

```python
target_fraction
# looks good so far!
```

Now let's get into the business of doing the importance samples and re-weights

```python

```

```python

```

```python
# Now we construct the weights matrix: how many sims are we planning for each value of theta?
n_per_thetaj = 1000
# We want this to net out to, let's say, 1% of the total weight...
sum_rejects = np.delete(target_fraction, 0, axis = 0)
any_successes = np.sum(sum_rejects, axis = 0) > 0
any_successes
relevant_mu = mu[any_successes]
```

```python
temp = sum_rejects / np.sum(sum_rejects,axis = 0)[None,:]
wjj = 0.01
important_weights = temp[:,any_successes]*(1 - wjj)
important_weights.shape
```

```python


```

```python
[relevant_mu.shape , important_weights.shape]
```

```python
np.diag(np.full_like(relevant_mu,wjj)).shape
```

```python
full_weights = np.append(np.diag(np.full_like(relevant_mu,wjj)), important_weights, axis = 0)
np.sum(full_weights, axis = 0)

```

```python
# Now we do the importance samples:
mu_importance = np.append(relevant_mu,mu_cluster_centers)
z_importance = np.random.normal(0, 1, size = 1000 * len(mu_importance)).reshape(1000, len(mu_importance))
data_importance = mu_importance[None,:] + z_importance
```

```python
data_importance.shape, mu_importance.shape, relevant_mu.shape, np.transpose(full_weights).shape
```

```python
#dimensions: i (simulations), j (initial theta), k (importance theta which generates samples), m (second dummy copy of importance theta)
inside_exponent = data_importance[:,None,:, None]*(mu_importance[None,None,None, :] - relevant_mu[None,:,None, None]) - mu_importance[None,None,None,:]**2/2 + relevant_mu[None,:,None, None]**2/2
likelihood_ratios = np.exp(inside_exponent) # I bet this is the problem! Look in this line and the above for a bug!
denoms = np.sum(likelihood_ratios * np.transpose(full_weights)[None, :, None, :], axis = 3)
rejects = data_importance > 1.96
inner_mean = np.mean(rejects[:,None,:]/denoms, axis = 0) #this is the inner sum divided by n_j
inner_mse_estimate = np.mean((rejects[:,None,:]/denoms)**2, axis = 0) - inner_mean**2 # trying to do an empirical calculation of the variance of each obs
final_result = np.sum(inner_mean * np.transpose(full_weights), axis = 1)
final_variance_estimate =np.sum((inner_mse_estimate/1000) * (np.transpose(full_weights)**2), axis = 1)
```

```python
final_result
```

```python
final_variance_estimate
```

```python
#estimated sample size ratio
((final_result * (1-final_result)) / (1000)) /final_variance_estimate
# Hmm... this is not excellent. Importance sampling is doing better for the first few values of mu, but not consistently for larger values.
# I'm betting that this is due to an incorrect variance calculation

```

```python
mu_importance
```

The denominator formula:

denom = sum w_jk Pk/Pj (X).

The likelihood ratio is exp([x -\ mu_j]^2/2 - [x-\mu_k]^2/2) = exp(-mu_k^2/2 + mu_j^2/2 + x(mu_k - mu_j))


Now let's generalize this to two-dimensional mu!

```python
mu = np.linspace(-2, 2, 11)
z = np.random.normal(0, 1, 1000)
data = mu[None,:] + z[:,None]
flat_data =  data.flatten()
#Now, we select the rejections in order to run k-means on them
selection = flat_data > 1.96
rejections = flat_data[selection]
standardized_rejections = (rejections - np.mean(rejections))/np.std(rejections)
n_clusters = 5
kmeans= KMeans(n_clusters=n_clusters,
               init = "random",
            n_init=10,
               max_iter=300,
               random_state = 42)
kmeans.fit(standardized_rejections.reshape(-1,1))
mu_cluster_centers = kmeans.cluster_centers_ * np.std(rejections) + np.mean(rejections)
kmeans.labels_
n_orig = len(flat_data)
flat_labels = np.full(n_orig, -1,dtype = np.int32)
flat_labels[selection] = kmeans.labels_
labels = flat_labels.reshape(data.shape)
n_sims_per_theta = data.shape[0] # 1000 for now
n_theta = data.shape[1] # 11 for now
target_fraction = np.full((n_clusters + 1, n_theta),-1)
labelset = np.unique(labels)
labelbins = np.append(labelset - 0.5,n_clusters - 0.5)
for i in range(n_theta):
    target_fraction[:,i] = np.histogram(labels[:,i], bins = labelbins)[0]
```

```python
#Pilot sims done, now the real run:
n_per_thetaj = 1000
# We want this to net out to, let's say, 1% of the total weight...
sum_rejects = np.delete(target_fraction, 0, axis = 0)
any_successes = np.sum(sum_rejects, axis = 0) > 0
any_successes
relevant_mu = mu[any_successes]
temp = sum_rejects / np.sum(sum_rejects,axis = 0)[None,:]
wjj = 0.01
important_weights = temp[:,any_successes]*(1 - wjj)
full_weights = np.append(np.diag(np.full_like(relevant_mu,wjj)), important_weights, axis = 0)
# Now we do the importance samples:
mu_importance = np.append(relevant_mu,mu_cluster_centers)
z_importance = np.random.normal(0, 1, size = 1000 * len(mu_importance)).reshape(1000, len(mu_importance))
data_importance = mu_importance[None,:] + z_importance
inside_exponent = data_importance[:,None,:, None]*(mu_importance[None,None,None, :] - relevant_mu[None,:,None, None]) - mu_importance[None,None,None,:]**2/2 + relevant_mu[None,:,None, None]**2/2
likelihood_ratios = np.exp(inside_exponent) # I bet this is the problem! Look in this line and the above for a bug!
denoms = np.sum(likelihood_ratios * np.transpose(full_weights)[None, :, None, :], axis = 3)
rejects = data_importance > 1.96
inner_mean = np.mean(rejects[:,None,:]/denoms, axis = 0) #this is the inner sum divided by n_j
inner_mse_estimate = np.mean((rejects[:,None,:]/denoms)**2, axis = 0) - inner_mean**2 # trying to do an empirical calculation of the variance of each obs
final_result = np.sum(inner_mean * np.transpose(full_weights), axis = 1)
final_variance_estimate =np.sum((inner_mse_estimate/1000) * (np.transpose(full_weights)**2), axis = 1)

```

```python
final_result
```
