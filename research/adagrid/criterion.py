import numpy as np

import confirm.mini_imprint.lewis_drivers as lts


class Criterion:
    def __init__(
        self,
        lei_obj,
        P,
        S,
        D,
    ):
        self.lei_obj = lei_obj

        ########################################
        # Criterion step 0: prep some useful numbers
        ########################################
        self.alpha_cost = P.alpha_target - S.alpha0
        twb_worst_lam = np.min(S.twb_min_lam)
        eps_twb = 1e-6
        ties = np.where(np.abs(S.twb_min_lam - twb_worst_lam) < eps_twb)[0]
        self.twb_worst_tile = np.argmin(S.twb_max_lam[ties])
        self.overall_tile = np.argmin(S.orig_lam)
        self.overall_lam = S.orig_lam[self.overall_tile]

        ########################################
        # Criterion step 1: is tuning impossible?
        ########################################
        # try to estimate the number of refinements steps required to get to the
        # target alpha. for now, it's okay to slightly preference refinement over
        # adding sims because refinment gives more information in a sense.
        self.cost_to_refine = 2**lei_obj.n_arms
        self.sims_required_to_rej_once = 2 / S.alpha0 - 1
        self.cost_to_rej_once = self.sims_required_to_rej_once / S.sim_sizes

        # if a tile always stops early, it's probably not interesting and we should
        # lean towards simulating more rather than more expensive refinement
        self.normally_stops_early = S.twb_mean_lam >= 1
        self.prefer_simulation = (self.cost_to_refine > self.cost_to_rej_once) & (
            self.normally_stops_early
        )

        self.alpha_to_rej_once = 2 / (S.sim_sizes + 1)
        self.impossible = S.alpha0 < self.alpha_to_rej_once
        self.impossible_refine = (self.impossible & (~self.prefer_simulation)) | (
            S.alpha0 == 0
        )
        self.impossible_sim = self.impossible & self.prefer_simulation

        ########################################
        # Criterion step 2: what is the bias?
        ########################################
        self.B_lamss_idx = S.B_lam.argmin(axis=0)
        self.B_lamss = S.B_lam[self.B_lamss_idx, np.arange(S.B_lam.shape[1])]
        self.bootstrap_min_lams = np.concatenate(
            ([self.overall_lam], np.min(S.B_lam, axis=0))
        )
        self.lam_std = self.bootstrap_min_lams.std()
        self.overall_stats = lts.one_stat(
            lei_obj,
            S.g.theta_tiles[self.overall_tile],
            S.g.null_truth[self.overall_tile],
            S.sim_sizes[self.overall_tile],
            D.unifs,
            D.unifs_order,
        )
        self.overall_typeI_sum = (
            self.overall_stats[None, :] < self.bootstrap_min_lams[:, None]
        ).sum(axis=1)
        self.bias = (
            self.overall_typeI_sum[0] - self.overall_typeI_sum[1:].mean()
        ) / S.sim_sizes[self.overall_tile]

        ########################################
        # Criterion step 3: Refine tiles that are too large, deepen tiles that
        # cause too much bias.
        ########################################
        self.which_deepen = np.zeros(S.g.n_tiles, dtype=bool)
        self.which_refine = np.zeros(S.g.n_tiles, dtype=bool)

        if (self.alpha_cost[self.overall_tile] < P.grid_target) and (
            self.bias < P.bias_target
        ):
            return

        self.inflation_factor = 2
        self.inflate_from = S.twb_max_lam[self.twb_worst_tile]
        self.inflated_min_lam = (
            self.inflate_from
            + (S.twb_min_lam - self.inflate_from) * self.inflation_factor
        )
        # this could be improved??
        # think about twb_max_lam[worst_tile] and twb_min_lam??
        self.focus = S.twb_min_lam < 1
        self.sorted_bootstrap_idxs = np.argsort(self.inflated_min_lam[self.focus])
        self.dangerous = np.where(self.focus)[0][
            self.sorted_bootstrap_idxs[: P.step_size]
        ]

        self.d_should_refine = self.alpha_cost[self.dangerous] > P.grid_target
        self.deepen_likely_to_work = (
            S.twb_mean_lam[self.dangerous] > S.twb_max_lam[self.twb_worst_tile]
        )
        self.d_should_deepen = self.deepen_likely_to_work & (
            S.sim_sizes[self.dangerous] < P.max_sim_size
        )
        self.which_refine[self.dangerous] = self.d_should_refine & (
            ~self.d_should_deepen
        )
        self.which_deepen[self.dangerous] = self.d_should_deepen | (
            self.bias > P.bias_target
        )

        self.which_refine |= self.impossible_refine
        self.which_deepen |= self.impossible_sim
        self.which_deepen &= ~self.which_refine
        self.which_deepen &= S.sim_sizes < P.max_sim_size

        self.report = dict(
            overall_lam=f"{self.overall_lam:.5f}",
            lam_std=f"{self.lam_std:.4f}",
            grid_cost=f"{self.alpha_cost[self.overall_tile]:.5f}",
            bias=f"{self.bias:.5f}",
            total_slack=f"{self.alpha_cost[self.overall_tile] + self.bias:.5f}",
            n_tiles=S.g.n_tiles,
            n_refine=np.sum(self.which_refine),
            n_refine_impossible=np.sum(self.impossible_refine),
            n_moresims=np.sum(self.which_deepen),
            n_moresims_impossible=np.sum(self.impossible_sim),
        )
