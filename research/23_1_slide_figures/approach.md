```python
from imprint.nb_util import setup_nb
setup_nb()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
```

```python
fig = plt.figure()
ax = fig.add_subplot(111)
rect1 = matplotlib.patches.Rectangle((0, 0), 1, 1, alpha=0.5, facecolor='none', edgecolor='black')
ax.add_patch(rect1)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.title('Null Space $\Theta$')
plt.axis('off')
plt.savefig('figures/approach_1.pdf', bbox_inches='tight')
plt.show()
```

```python
fig = plt.figure()
ax = fig.add_subplot(111)
for yi, ya in enumerate([0.5, 0]):
    for xi, xa in enumerate([0, 0.5]):
        width = 0.5
        rect = matplotlib.patches.Rectangle((xa, ya), width, width, alpha=0.5, facecolor='none', edgecolor='black')
        ax.add_patch(rect)
        xp = xa + width / 2
        yp = ya + width / 2 
        ax.scatter(xp, yp, s=20, c='black')
        eps = 2e-2
        i = 2 * yi + xi + 1
        ax.annotate(f'$\\theta_{i}$', (xp + eps, yp + eps))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.title('Null Space $\Theta$')
plt.axis('off')
plt.savefig('figures/approach_2.pdf', bbox_inches='tight')
plt.show()
```

```python
fig = plt.figure()
ax = fig.add_subplot(111)
for yi, ya in enumerate([0.5, 0]):
    for xi, xa in enumerate([0, 0.5]):
        width = 0.5
        rect = matplotlib.patches.Rectangle((xa, ya), width, width, alpha=0.5, facecolor='none', edgecolor='black')
        ax.add_patch(rect)
        xp = xa + width / 2
        yp = ya + width / 2 
        ax.scatter(xp, yp, s=50, c='green', marker='*')
        eps = 2e-2
        i = 2 * yi + xi + 1
        ax.annotate(f'$\\theta_{i}$', (xp + eps, yp + eps))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.title('Null Space $\Theta$')
plt.axis('off')
plt.savefig('figures/approach_3.pdf', bbox_inches='tight')
plt.show()
```

```python
fig = plt.figure()
ax = fig.add_subplot(111)
for yi, ya in enumerate([0.5, 0]):
    for xi, xa in enumerate([0, 0.5]):
        width = 0.5
        color = 'green' if xa == 0 and ya == 0.5 else 'none'
        rect = matplotlib.patches.Rectangle((xa, ya), width, width, alpha=0.2, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        xp = xa + width / 2
        yp = ya + width / 2 
        ax.scatter(xp, yp, s=50, c='green', marker='*')
        eps = 2e-2
        i = 2 * yi + xi + 1
        ax.annotate(f'$\\theta_{i}$', (xp + eps, yp + eps))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.title('Null Space $\Theta$')
plt.axis('off')
plt.savefig('figures/approach_4.pdf', bbox_inches='tight')
plt.show()
```

```python
fig = plt.figure()
ax = fig.add_subplot(111)
for yi, ya in enumerate([0.5, 0]):
    for xi, xa in enumerate([0, 0.5]):
        width = 0.5
        rect = matplotlib.patches.Rectangle((xa, ya), width, width, alpha=0.2, facecolor='green', edgecolor='black')
        ax.add_patch(rect)
        xp = xa + width / 2
        yp = ya + width / 2 
        ax.scatter(xp, yp, s=50, c='green', marker='*')
        eps = 2e-2
        i = 2 * yi + xi + 1
        ax.annotate(f'$\\theta_{i}$', (xp + eps, yp + eps))
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.title('Null Space $\Theta$')
plt.axis('off')
plt.savefig('figures/approach_5.pdf', bbox_inches='tight')
plt.show()
```
