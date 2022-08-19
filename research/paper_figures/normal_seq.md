---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3.10.5 ('confirm')
    language: python
    name: python3
---

```python
import outlaw.nb_util as nb_util
nb_util.setup_nb()
```

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
```

```python
npts = 3
mu = np.linspace(-1, 0, npts)
stepsize = mu[1] - mu[0]
mu = mu - stepsize/2
power = 1 - scipy.stats.norm.cdf(- mu + 1.96)
plt.plot(mu, power, 'o')
plt.show()
```

```python

```
