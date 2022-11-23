```python
from confirm.outlaw.nb_util import setup_nb

setup_nb()

import confirm.imprint as ip
from confirm.models.binom1d import Binom1D
```

```python
g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
rej_df = ip.validate(Binom1D, g, 0.5, K=2**20, model_kwargs={"n": 100})
rej_df
```

```python
rej_df
```
