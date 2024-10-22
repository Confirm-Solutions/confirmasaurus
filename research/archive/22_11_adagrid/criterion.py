import numpy as np

import confirm.adagrid.lewis_drivers as ld


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
        eps_twb = 1e-6
        self.alpha_cost = P.alpha_target - S.alpha0

        self.twb_worst_lam = np.min(S.twb_max_lam)
        self.ties = np.where(np.abs(S.twb_max_lam - self.twb_worst_lam) < eps_twb)[0]
        self.twb_worst_tile = self.ties[np.argmin(S.twb_min_lam[self.ties])]
        self.overall_tile = np.argmin(S.orig_lam)
        self.overall_lam = S.orig_lam[self.overall_tile]

        idxs = [self.twb_worst_tile]
        self.twb_worst_tile_lams = ld.bootstrap_tune_runner(
            self.lei_obj,
            S.sim_sizes[idxs],
            np.full(1, P.alpha_target),
            S.g.theta_tiles[idxs],
            S.g.null_truth[idxs],
            D.unifs,
            D.bootstrap_idxs,
            D.unifs_order,
            sim_batch_size=1024 * 16,
            grid_batch_size=1,
        )
        self.twb_worst_tile_lam_min = self.twb_worst_tile_lams[
            0, 1 + P.nB_global :
        ].min()
        self.twb_worst_tile_lam_mean = self.twb_worst_tile_lams[
            0, 1 + P.nB_global :
        ].mean()
        self.twb_worst_tile_lam_max = self.twb_worst_tile_lams[
            0, 1 + P.nB_global :
        ].max()

        ########################################
        # Criterion step 1: is tuning impossible?
        ########################################

        # TODO: the impossibility condition only needs to be checked for new
        # tiles that used to be impossible (or simplifying: new tiles). once a
        # tile is not impossible, it will never again be impossible.

        self.cost_to_refine = 2**lei_obj.n_arms
        self.sims_to_rej_enough = (P.tuning_min_idx + 1) / S.alpha0 - 1
        self.alpha_to_rej_enough = (P.tuning_min_idx + 1) / (S.sim_sizes + 1)
        self.cost_to_rej_once = self.sims_to_rej_enough / S.sim_sizes

        # if a tile always stops early, it's probably not interesting and we should
        # lean towards simulating more rather than more expensive refinement
        self.normally_stops_early = S.twb_mean_lam >= 1
        self.prefer_simulation = (self.cost_to_refine > self.cost_to_rej_once) & (
            self.normally_stops_early
        )

        self.impossible = S.alpha0 < self.alpha_to_rej_enough
        self.impossible_refine_orig = (self.impossible & (~self.prefer_simulation)) | (
            S.alpha0 == 0
        )
        self.impossible_refine = np.zeros(S.g.n_tiles, dtype=bool)
        self.impossible_refine[
            np.where(self.impossible_refine_orig)[0][: P.step_size]
        ] = True
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
        self.overall_stats = ld.one_stat(
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
        # Criterion step 3: Refine and deepen based on inflated bootstrapped
        # minimum lambda
        ########################################
        self.orderer = S.twb_mean_lam + (S.twb_min_lam - S.twb_mean_lam) * 4
        ignore = S.twb_mean_lam > 0.2
        self.orderer[ignore] = S.twb_min_lam[ignore]
        self.sorted_ordering = np.argsort(self.orderer)
        self.sorted_orderer = self.orderer[self.sorted_ordering]

        self.dangerous = self.sorted_ordering[: P.step_size]

        self.d_should_refine = self.alpha_cost[self.dangerous] > P.grid_target
        self.deepen_likely_to_work = (
            S.twb_mean_lam[self.dangerous] > self.twb_worst_tile_lam_mean
        )
        self.d_should_deepen = (
            self.deepen_likely_to_work & (S.sim_sizes[self.dangerous] < P.max_sim_size)
        ) | (~self.d_should_refine)

        ########################################
        # Criterion step 4: Refine based on bootstrapped mean lambda
        ########################################
        self.refine_orderer = S.twb_mean_lam
        self.sorted_refine_ordering = np.argsort(self.refine_orderer)
        self.sorted_refine_orderer = self.refine_orderer[self.sorted_refine_ordering]
        self.refine_dangerous = self.sorted_refine_ordering[: P.step_size // 2]
        self.d_should_refine2 = self.alpha_cost[self.refine_dangerous] > P.grid_target

        ########################################
        # Criterion step 5: Actually assign refine/deepen decisions.
        ########################################
        self.which_deepen = np.zeros(S.g.n_tiles, dtype=bool)
        self.which_refine = np.zeros(S.g.n_tiles, dtype=bool)
        if (self.alpha_cost[self.overall_tile] > P.grid_target) or (
            self.bias > P.bias_target
        ):
            self.which_refine[self.refine_dangerous] = self.d_should_refine2
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

        ########################################
        # Criterion step 6: Reporting
        ########################################
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
            twb_worst_tile_lam_min=f"{self.twb_worst_tile_lam_min:.5f}",
            twb_worst_tile_lam_mean=f"{self.twb_worst_tile_lam_mean:.5f}",
            twb_worst_tile_lam_max=f"{self.twb_worst_tile_lam_max:.5f}",
        )
        self.report["min(twb_min_lam)"] = f"{S.twb_min_lam.min():.5f}"
        self.report["min(twb_mean_lam)"] = f"{S.twb_mean_lam.min():.5f}"
        self.report["min(twb_max_lam)"] = f"{S.twb_max_lam.min():.5f}"
        self.report["orderer < min(twb_mean_lam)"] = np.sum(
            self.orderer < S.twb_mean_lam.min()
        )
        self.report["orderer < min(twb_max_lam)"] = np.sum(
            self.orderer < S.twb_max_lam.min()
        )
        self.report[
            "max(twb_min_lam[dangerous])"
        ] = f"{S.twb_min_lam[self.dangerous].max():.5f}"

        query = self.orderer[self.overall_tile]
        self.report["overall priority"] = np.searchsorted(self.sorted_orderer, query)
        self.report["min(B_lamss)"] = np.min(self.B_lamss)
        query = self.orderer[self.B_lamss_idx[self.B_lamss.argmin()]]
        self.report["min(B_lamss) priority"] = np.searchsorted(
            self.sorted_orderer, query
        )
