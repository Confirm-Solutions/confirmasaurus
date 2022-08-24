import jax.numpy as jnp


class BaseTable:
    def __init__(self, n_sizes):
        # compute mask to hash n_sizes
        n_arms = n_sizes.shape[-1]
        n_sizes_max = jnp.max(n_sizes) + 1
        n_sizes_max_mask = n_sizes_max ** jnp.arange(0, n_arms)
        self.n_sizes_max_mask = n_sizes_max_mask.astype(int)

        # create hashes
        hashes = jnp.array([self.hash_n__(ns) for ns in n_sizes])

        # reorder data based on increasing order of hashes
        self.hashes_order = jnp.argsort(hashes)
        self.hashes = hashes[self.hashes_order]

    def hash_n__(self, n):
        """
        Hashes the n configuration with a given mask.

        Parameters:
        -----------
        n:      n configuration sorted in decreasing order.
        """
        return jnp.sum(n * self.n_sizes_max_mask)

    def search(self, n):
        n_hash = self.hash_n__(n)
        idx = jnp.searchsorted(self.hashes, n_hash)
        return idx

    def hash_ordered(self, seq):
        return tuple(seq[i] for i in self.hashes_order)
