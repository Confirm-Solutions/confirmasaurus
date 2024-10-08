"""
The only thing better than WD40 is WD41.
"""
import jax
import jax.numpy as jnp
import numpy as np

import imprint as ip

default_frac_tnbc = 0.54


class WD41Null(ip.grid.NullHypothesis):
    def __init__(self, frac_tnbc=default_frac_tnbc):
        self.frac_tnbc = frac_tnbc
        self.jit_f = jax.jit(jax.vmap(self.f))

    def get_theta(self, theta):
        return theta

    def f(self, theta):
        (
            pcontrol_tnbc,
            ptreat_tnbc,
            pcontrol_hrplus,
            ptreat_hrplus,
        ) = jax.scipy.special.expit(self.get_theta(theta))
        control_term = (
            self.frac_tnbc * pcontrol_tnbc + (1 - self.frac_tnbc) * pcontrol_hrplus
        )
        treat_term = self.frac_tnbc * ptreat_tnbc + (1 - self.frac_tnbc) * ptreat_hrplus
        return control_term - treat_term

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
        frac_tnbc=default_frac_tnbc,
        ignore_intersection=False,
        dtype=jnp.float32,
    ):
        self.dtype = dtype
        self.ignore_intersection = ignore_intersection

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
            key, (max_K, self.n_first_stage + self.n_second_stage)
        ).astype(self.dtype)

        self.sim_stat_jit = jax.vmap(
            jax.vmap(jax.jit(self.sim_stat), in_axes=(0, None, None)),
            in_axes=(None, 0, 0),
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

    def sample(self, unifs, p):
        return jnp.sum(unifs < p, dtype=self.dtype)

    def sample_all_arms(
        self,
        unifs_stage,
        pcontrol_tnbc,
        ptreat_tnbc,
        pcontrol_hrplus,
        ptreat_hrplus,
        n_tnbc_per_arm,
        n_hrplus_per_arm,
    ):
        unifs_tnbc = unifs_stage[: self.n_tnbc_second_stage_total]
        unifs_tnbc_control = unifs_tnbc[:n_tnbc_per_arm]
        unifs_tnbc_treat = unifs_tnbc[n_tnbc_per_arm:]
        unifs_hrplus = unifs_stage[2 * (n_tnbc_per_arm) :]
        unifs_hrplus_control = unifs_hrplus[:n_hrplus_per_arm]
        unifs_hrplus_treat = unifs_hrplus[n_hrplus_per_arm:]

        ytnbccontrol = self.sample(unifs_tnbc_control, pcontrol_tnbc)
        ytnbctreat = self.sample(unifs_tnbc_treat, ptreat_tnbc)
        yhrpluscontrol = self.sample(unifs_hrplus_control, pcontrol_hrplus)
        yhrplustreat = self.sample(unifs_hrplus_treat, ptreat_hrplus)
        return ytnbccontrol, ytnbctreat, yhrpluscontrol, yhrplustreat

    def zstat(self, y_control, y_treat, n_per_arm):
        y_avg = (y_treat + y_control) * 0.5
        denominatortnbc = jnp.sqrt(y_avg * (n_per_arm - y_avg) * (2 / n_per_arm))
        return (y_treat - y_control) / denominatortnbc

    def stage1(self, unifs, pcontrol_tnbc, ptreat_tnbc, pcontrol_hrplus, ptreat_hrplus):
        (
            ytnbccontrol,
            ytnbctreat,
            yhrpluscontrol,
            yhrplustreat,
        ) = self.sample_all_arms(
            unifs,
            pcontrol_tnbc,
            ptreat_tnbc,
            pcontrol_hrplus,
            ptreat_hrplus,
            self.n_tnbc_first_stage_per_arm,
            self.n_hrplus_first_stage_per_arm,
        )

        # Arm-dropping logic: drop all elementary hypotheses with larger than
        # 0.1 difference in effect size
        tnbc_effect = (ytnbctreat - ytnbccontrol) / self.n_tnbc_first_stage_per_arm
        hrplus_effect = (
            yhrplustreat - yhrpluscontrol
        ) / self.n_hrplus_first_stage_per_arm
        full_effect = (
            self.true_frac_tnbc * tnbc_effect
            + (1 - self.true_frac_tnbc) * hrplus_effect
        )
        effect_diff = tnbc_effect - full_effect

        hypofull_live = effect_diff <= 0.1
        hypotnbc_live = effect_diff >= -0.1

        y_control = ytnbccontrol + yhrpluscontrol
        y_treat = ytnbctreat + yhrplustreat
        n_control = self.n_tnbc_first_stage_per_arm + self.n_hrplus_first_stage_per_arm
        return (
            self.zstat(ytnbccontrol, ytnbctreat, self.n_tnbc_first_stage_per_arm),
            self.zstat(y_control, y_treat, n_control),
            hypofull_live,
            hypotnbc_live,
        )

    def stage2_tnbc(self, unifs_stage, pcontrol_tnbc, ptreat_tnbc):
        # Here, we ignored n_hrplus_second_stage_per_arm patients because the
        # hrplus arm has been dropped.
        unifs_control = unifs_stage[: self.n_tnbc_only_second_stage]
        unifs_treat = unifs_stage[self.n_tnbc_only_second_stage :]
        ytnbccontrol = self.sample(unifs_control, pcontrol_tnbc)
        ytnbctreat = self.sample(unifs_treat, ptreat_tnbc)
        return (
            self.zstat(ytnbccontrol, ytnbctreat, self.n_tnbc_only_second_stage),
            -self.dtype(jnp.inf),
        )

    def stage2_full(
        self,
        unifs_stage,
        pcontrol_tnbc,
        ptreat_tnbc,
        pcontrol_hrplus,
        ptreat_hrplus,
    ):
        (
            ytnbccontrol,
            ytnbctreat,
            yhrpluscontrol,
            yhrplustreat,
        ) = self.sample_all_arms(
            unifs_stage,
            pcontrol_tnbc,
            ptreat_tnbc,
            pcontrol_hrplus,
            ptreat_hrplus,
            self.n_tnbc_second_stage_per_arm,
            self.n_hrplus_second_stage_per_arm,
        )
        y_control = ytnbccontrol + yhrpluscontrol
        y_treat = ytnbctreat + yhrplustreat
        n_control = (
            self.n_tnbc_second_stage_per_arm + self.n_hrplus_second_stage_per_arm
        )
        return (
            self.zstat(ytnbccontrol, ytnbctreat, self.n_tnbc_second_stage_per_arm),
            self.zstat(y_control, y_treat, n_control),
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
        invsqrt2 = 1.0 / jnp.sqrt(2)
        hyptnbc_zstat = (ztnbc_stage1 + ztnbc_stage2) * invsqrt2
        hypfull_zstat = (zfull_stage1 + zfull_stage2) * invsqrt2

        if self.ignore_intersection:
            tnbc_stat = jax.scipy.stats.norm.cdf(-hyptnbc_zstat)
            full_stat = jax.scipy.stats.norm.cdf(-hypfull_zstat)
            if detailed:
                return dict(
                    hypotnbc_live=hypotnbc_live,
                    hypofull_live=hypofull_live,
                    hyptnbc_zstat=hyptnbc_zstat,
                    hypfull_zstat=hypfull_zstat,
                    tnbc_stat=tnbc_stat,
                    full_stat=full_stat,
                    ztnbc_stage1=ztnbc_stage1,
                    ztnbc_stage2=ztnbc_stage2,
                    zfull_stage1=zfull_stage1,
                    zfull_stage2=zfull_stage2,
                )
        else:
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
                        - jax.scipy.stats.norm.cdf(
                            jnp.maximum(ztnbc_stage2, zfull_stage2)
                        )
                    )
                ),
                jnp.where(hypotnbc_live, ztnbc_stage2, zfull_stage2),
            )

            HI_zcombined = (HI_zfirst + HI_zsecond) * invsqrt2

            tnbc_stat = jax.scipy.stats.norm.cdf(
                -jnp.minimum(hyptnbc_zstat, HI_zcombined)
            )
            full_stat = jax.scipy.stats.norm.cdf(
                -jnp.minimum(hypfull_zstat, HI_zcombined)
            )
            if detailed:
                return dict(
                    hypotnbc_live=hypotnbc_live,
                    hypofull_live=hypofull_live,
                    hyptnbc_zstat=hyptnbc_zstat,
                    hypfull_zstat=hypfull_zstat,
                    HI_zcombined=HI_zcombined,
                    tnbc_stat=tnbc_stat,
                    full_stat=full_stat,
                    ztnbc_stage1=ztnbc_stage1,
                    ztnbc_stage2=ztnbc_stage2,
                    zfull_stage1=zfull_stage1,
                    zfull_stage2=zfull_stage2,
                )

        return tnbc_stat, full_stat

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

    def sim_stat(self, unifs, theta, null_truth):
        p = jax.scipy.special.expit(theta)
        tnbc_stat, full_stat = self.sim(
            unifs, p[..., 0], p[..., 1], p[..., 2], p[..., 3], detailed=False
        )
        rejected = jnp.array((tnbc_stat, full_stat))
        stat = jnp.min(jnp.where(null_truth, rejected, 1e9))
        return stat

    def sim_batch(
        self,
        begin_sim: int,
        end_sim: int,
        theta: jnp.ndarray,
        null_truth: jnp.ndarray,
        detailed: bool = False,
    ):
        out = self.sim_stat_jit(
            self.unifs[begin_sim:end_sim],
            theta.astype(self.dtype),
            null_truth.astype(self.dtype),
        )
        # assert out.dtype == self.dtype
        return out
