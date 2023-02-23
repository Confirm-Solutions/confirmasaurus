"""
The only thing better than WD40 is WD41.
"""
import jax
import jax.numpy as jnp
import numpy as np

import imprint as ip


class WD41Null(ip.grid.NullHypothesis):
    def __init__(self, frac_tnbc):
        self.frac_tnbc = frac_tnbc

        def f(theta):
            (
                pcontrol_tnbc,
                ptreat_tnbc,
                pcontrol_hrplus,
                ptreat_hrplus,
            ) = jax.scipy.special.expit(theta)
            control_term = frac_tnbc * pcontrol_tnbc + (1 - frac_tnbc) * pcontrol_hrplus
            treat_term = frac_tnbc * ptreat_tnbc + (1 - frac_tnbc) * ptreat_hrplus
            return control_term - treat_term

        self.jit_f = jax.jit(jax.vmap(f))

    def dist(self, theta):
        return self.jit_f(theta)

    def description(self):
        return f"WD41Null({self.frac_tnbc})"


class WD41(ip.Model):
    """
    theta[0] = logit(pcontrol_tnbc)
    theta[1] = logit(ptreat_tnbc)
    theta[2] = logit(pcontrol_hrplus)
    theta[3] = logit(ptreat_hrplus)
    """

    def __init__(
        self,
        seed,
        max_K,
        n_requested_first_stage=150,
        n_requested_second_stage=150,
        frac_tnbc=0.54,
        dtype=jnp.float32,
    ):
        def split(n_stage):
            return (
                int(np.floor((n_stage * frac_tnbc) * 0.5)),
                int(np.floor(n_stage * (1 - frac_tnbc) * 0.5)),
            )

        self.requested_frac_tnbc = frac_tnbc

        self.n_tnbc_first_stage_per_arm, self.n_hrplus_first_stage_per_arm = split(
            n_requested_first_stage
        )
        self.n_tnbc_first_stage_total = self.n_tnbc_first_stage_per_arm * 2
        self.n_hrplus_first_stage_total = self.n_hrplus_first_stage_per_arm * 2
        self.n_first_stage = (
            self.n_tnbc_first_stage_total + self.n_hrplus_first_stage_total
        )

        self.n_tnbc_second_stage_per_arm, self.n_hrplus_second_stage_per_arm = split(
            n_requested_second_stage
        )
        self.n_tnbc_second_stage_total = self.n_tnbc_second_stage_per_arm * 2
        self.n_hrplus_second_stage_total = self.n_hrplus_second_stage_per_arm * 2
        self.n_second_stage = (
            self.n_tnbc_second_stage_total + self.n_hrplus_second_stage_total
        )

        self.n_tnbc_only_second_stage = self.n_second_stage // 2

        self.true_frac_tnbc = self.n_tnbc_first_stage_total / self.n_first_stage

        key = jax.random.PRNGKey(0)
        self.unifs = jax.random.uniform(
            key, (max_K, self.n_first_stage + self.n_second_stage), dtype=dtype
        )
        self.sim_jit = jax.vmap(
            jax.vmap(
                jax.jit(self.sim, static_argnames=("detailed",)),
                in_axes=(0, None, None, None, None),
            ),
            in_axes=(None, 0, 0, 0, 0),
        )

        self.null_hypos = [
            ip.hypo("theta0 > theta1"),
            WD41Null(self.true_frac_tnbc),
        ]
        self.family = "binomial"
        self.n_max_tnbc = (
            self.n_tnbc_first_stage_per_arm + self.n_tnbc_only_second_stage
        )
        self.n_max_hrplus = (
            self.n_hrplus_first_stage_per_arm + self.n_hrplus_second_stage_per_arm
        )
        self.family_params = {
            "n": jnp.array(
                [
                    self.n_max_tnbc,
                    self.n_max_tnbc,
                    self.n_max_hrplus,
                    self.n_max_hrplus,
                ]
            )
        }

    def sample(self, unifs, start, end, p):
        return jnp.sum(unifs[start:end] < p) / (end - start)

    def sample_all_arms(
        self, unifs, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus
    ):
        unifs_tnbc = unifs[: self.n_tnbc_second_stage_total]
        phattnbccontrol = self.sample(
            unifs_tnbc, 0, self.n_tnbc_second_stage_per_arm, pcontrol_tnbc
        )
        phattnbctreat = self.sample(
            unifs_tnbc,
            self.n_tnbc_second_stage_per_arm,
            self.n_tnbc_second_stage_total,
            ptreat_tnbc,
        )
        unifs_hrplus = unifs[self.n_tnbc_second_stage_total :]
        phathrpluscontrol = self.sample(
            unifs_hrplus, 0, self.n_hrplus_second_stage_per_arm, pcontrol_hrplus
        )
        phathrplustreat = self.sample(
            unifs_hrplus,
            self.n_hrplus_second_stage_per_arm,
            self.n_hrplus_second_stage_total,
            ptreat_hrplus,
        )
        return phattnbccontrol, phattnbctreat, phathrpluscontrol, phathrplustreat

    def ztnbc(self, phattnbccontrol, phattnbctreat, n_per_arm):
        tnbc_pooledaverage = (phattnbctreat + phattnbccontrol) * 0.5
        denominatortnbc = jnp.sqrt(
            tnbc_pooledaverage * (1 - tnbc_pooledaverage) * (2 / n_per_arm)
        )
        return (phattnbctreat - phattnbccontrol) / denominatortnbc

    def zfull(self, phattnbccontrol, phattnbctreat, phathrpluscontrol, phathrplustreat):
        totally_pooledaverage = (
            phattnbctreat + phattnbccontrol
        ) * self.n_tnbc_first_stage_per_arm / self.n_first_stage + (
            phathrplustreat + phathrpluscontrol
        ) * self.n_hrplus_first_stage_per_arm / self.n_first_stage
        denominatortotallypooled = jnp.sqrt(
            totally_pooledaverage
            * (1 - totally_pooledaverage)
            * (
                2
                / (self.n_tnbc_first_stage_per_arm + self.n_hrplus_first_stage_per_arm)
            )
        )
        tnbc_effect = phattnbctreat - phattnbccontrol
        hrplus_effect = phathrplustreat - phathrpluscontrol
        return (
            (tnbc_effect * self.n_tnbc_first_stage_total / self.n_first_stage)
            + (hrplus_effect * self.n_hrplus_first_stage_total / self.n_first_stage)
        ) / denominatortotallypooled

    def stage1(self, unifs, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus):
        (
            phattnbccontrol,
            phattnbctreat,
            phathrpluscontrol,
            phathrplustreat,
        ) = self.sample_all_arms(
            unifs, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus
        )

        # Arm-dropping logic: drop all elementary hypotheses with larger than
        # 0.1 difference in effect size
        tnbc_effect = phattnbctreat - phattnbccontrol
        hrplus_effect = phathrplustreat - phathrpluscontrol
        effectsize_difference = tnbc_effect - hrplus_effect
        # TODO: investigate this dropping logic. Section 3.4, ctrl-f "epsilon"
        # TODO: this is wrong and should be compared to the weighted average
        # treatment effects instead of the hrplus treatment effect??
        hypofull_live = effectsize_difference <= 0.1
        hypotnbc_live = effectsize_difference >= -0.1

        return (
            self.ztnbc(phattnbccontrol, phattnbctreat, self.n_tnbc_first_stage_per_arm),
            self.zfull(
                phattnbccontrol, phattnbctreat, phathrpluscontrol, phathrplustreat
            ),
            hypofull_live,
            hypotnbc_live,
        )

    def stage2_tnbc(self, unifs, pcontrol_tnbc, ptreat_tnbc):
        # Here, we ignored n_hrplus_second_stage_per_arm patients because the
        # hrplus arm has been dropped.
        phattnbccontrol = self.sample(
            unifs, 0, self.n_tnbc_only_second_stage, pcontrol_tnbc
        )
        phattnbctreat = self.sample(
            unifs,
            self.n_tnbc_only_second_stage,
            self.n_tnbc_only_second_stage * 2,
            ptreat_tnbc,
        )
        return (
            self.ztnbc(phattnbccontrol, phattnbctreat, self.n_tnbc_only_second_stage),
            -np.inf,
        )

    def stage2_full(
        self,
        unifs,
        pcontrol_tnbc,
        ptreat_tnbc,
        pcontrol_hrplus,
        ptreat_hrplus,
    ):
        (
            phattnbccontrol,
            phattnbctreat,
            phathrpluscontrol,
            phathrplustreat,
        ) = self.sample_all_arms(
            unifs, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus
        )
        return (
            self.ztnbc(
                phattnbccontrol, phattnbctreat, self.n_tnbc_second_stage_per_arm
            ),
            self.zfull(
                phattnbccontrol, phattnbctreat, phathrpluscontrol, phathrplustreat
            ),
        )

    def sim(
        self,
        unifs,
        pcontrol_tnbc=0.34,
        ptreat_tnbc=0.38,
        pcontrol_hrplus=0.23,
        ptreat_hrplus=0.27,
        detailed=False,
    ):
        unifs_stage1 = unifs[: self.n_first_stage]
        unifs_stage2 = unifs[self.n_first_stage :]

        ztnbc_stage1, zfull_stage1, hypofull_live, hypotnbc_live = self.stage1(
            unifs_stage1, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus
        )

        # now compute second-stage z-statistics depending on whether we drop the
        # hrplus arm or not
        ztnbc_stage2, zfull_stage2 = jax.lax.cond(
            hypofull_live,
            lambda: self.stage2_full(
                unifs_stage2,
                pcontrol_tnbc,
                ptreat_tnbc,
                pcontrol_hrplus,
                ptreat_hrplus,
            ),
            lambda: self.stage2_tnbc(unifs_stage2, pcontrol_tnbc, ptreat_tnbc),
        )

        # now combine test statistics
        # Now we go through the 3 intersection tests:
        hyptnbc_zstat = (ztnbc_stage1 + ztnbc_stage2) / jnp.sqrt(2)
        hypfull_zstat = (zfull_stage1 + zfull_stage2) / jnp.sqrt(2)

        # Now doing the combination rule for the intersection test
        # we multiply the p-value by two by analogy to bonferroni
        HI_pfirst = 2 * (
            1 - jax.scipy.stats.norm.cdf(jnp.maximum(ztnbc_stage1, zfull_stage1))
        )
        HI_zfirst = jax.scipy.stats.norm.ppf(1 - HI_pfirst)
        HI_zsecond = jnp.where(
            hypofull_live & hypotnbc_live,
            jax.scipy.stats.norm.ppf(
                1
                - 2
                * (
                    1
                    - jax.scipy.stats.norm.cdf(jnp.maximum(ztnbc_stage2, zfull_stage2))
                )
            ),
            jnp.where(hypotnbc_live, ztnbc_stage2, zfull_stage2),
        )

        HI_zcombined = (HI_zfirst + HI_zsecond) / jnp.sqrt(2)

        tnbc_stat = jax.scipy.stats.norm.cdf(-jnp.minimum(hyptnbc_zstat, HI_zcombined))
        full_stat = jax.scipy.stats.norm.cdf(-jnp.minimum(hypfull_zstat, HI_zcombined))

        # # Now we resolve which elementary statistics actually reject the null
        # hypothesis
        # rejectintersection = HI_zcombined > 1.96
        # rejecttnbc_elementary = hyptnbc_zstat > 1.96
        # rejectfull_elementary = hypfull_zstat > 1.96

        # rejecttnbc_final = (
        #     rejecttnbc_elementary & rejectintersection
        # )  # we use this for actual hypothesis rejections!
        # rejectfull_final = (
        #     rejectfull_elementary & rejectintersection
        # )  # we use this for actual hypothesis rejections!

        if detailed:
            return dict(
                hypotnbc_live=hypotnbc_live,
                hypofull_live=hypofull_live,
                tnbc_stat=tnbc_stat,
                full_stat=full_stat,
                ztnbc_stage1=ztnbc_stage1,
                ztnbc_stage2=ztnbc_stage2,
                zfull_stage1=zfull_stage1,
                zfull_stage2=zfull_stage2,
                HI_pfirst=HI_pfirst,
                HI_zfirst=HI_zfirst,
                HI_zsecond=HI_zsecond,
                HI_zcombined=HI_zcombined,
            )
        else:
            return tnbc_stat, full_stat

    def sim_batch(
        self,
        begin_sim: int,
        end_sim: int,
        theta: jnp.ndarray,
        null_truth: jnp.ndarray,
        detailed: bool = False,
    ):
        p = jax.scipy.special.expit(theta)
        subsample = self.unifs[begin_sim:end_sim]
        tnbc_stat, full_stat = self.sim_jit(
            subsample, p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        )
        rejected = np.stack((tnbc_stat, full_stat), axis=-1)
        stat = jnp.min(jnp.where(null_truth[:, None], rejected, 1e9), axis=-1)
        return stat
