# src.synthetic.create_life_policy.py
"""Create synthetic single life policy"""
# copyright 2024 Oreum OÜ

import ipywidgets as wgts
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display
from scipy import stats

all = ["LifePolicyBuilder"]

sns.set(
    style="darkgrid",
    palette="muted",
    context="notebook",
    rc={"savefig.dpi": 300, "figure.figsize": (12, 3)},
)


class LifePolicyBuilder:
    """Create synthetic single life policy with a lifetime"""

    # rng = np.random.default_rng(seed=42)

    def __init__(self):
        """Set some defaults"""
        self.ref_vals = dict(
            age_incept=40,
            age_max=121,  # max age by regulation, unlikely to change soon
            survival_shape=0.09,  # param value for linear component of weibull
            survival_scale=70,
            age_today=40,
            death_benefit=500000,
            db_covered_by=20,  # death ben covered by N years
            n_prem_payments=10,
            invest_rate=0.04,
            brokerage_ratio=0.2,  # paid out to originating broker
            cost_ratio=0.1,  # insurer operational costs apportioned
            profit_ratio=0.1,  # insurer pure profit
            n_years_frontload=3,  # insurer costs and profit
            churn=True,  # consider churn
            churn_shape=0.1,  # note 10x too big for convenience wgt input render
            churn_scale=30,
        )
        self._df = pd.DataFrame()  # see hack in plot_policy_lifetime()

    def create(self, ref_vals: dict = None) -> pd.DataFrame:
        """Create synthetic policy lifetime
        NOTE:
            + Policies are assumed to incept on e.g. 01 Jan, although calendar
              month-day doesnt actually exist: all values are quantized annually
              and applicable to end-of-this-year (eoty)
            + We accept age_incept, age_max, and age_today as current values
              but the point-in-time rows in the dataframe are values calculated
              as-at end-of-this-year (eoty) after all actions have been taken
        """
        if ref_vals is None:
            ref_vals = {}
        v = self.ref_vals.copy()
        v.update(**ref_vals)

        # ensure age concordances
        v["age_incept"] = min(v["age_incept"], v["age_max"] - v["n_prem_payments"])
        v["age_today"] = min(v["age_today"], v["age_max"])

        # create core df
        df = pd.DataFrame(
            dict(
                age_of_insured_eoty=np.arange(v["age_incept"] + 1, v["age_max"] + 1),
                year_inforce=np.arange(v["age_max"] - v["age_incept"]),
            )
        ).set_index("year_inforce")

        # static vals
        df["death_benefit"] = -v["death_benefit"]

        # basic generalised life survival curve
        # NOTE:
        #   + based on a CDF (with shape conditional on age)
        #       (1) Weibull prior params
        #       (2) Gompertz posterior means from NB300
        #   + principle based on https://arxiv.org/pdf/q-bio/0402013
        #   + param values adapted from paper to achieve zeros at 121
        #   + scipy params, using "_min" since x>0
        #       c == shape == k. when k=1 expon, k=2 Rayleigh, k>1 aging effect
        #       scale == lambda
        #   + Calc from age_incept to allow scaling (based on conditioning below)
        #     trim this before adding to df to start at end-of-year age_incept+1
        # surv = stats.weibull_min.sf(
        #     x=np.arange(v['age_incept'], v['age_max'] + 1),
        #     c=1.44 + v['survival_shape'] * v['age_incept'],
        #     loc=0,
        #     scale=v['survival_scale'],
        # )

        M = v[
            "survival_m"
        ]  # np.exp(3.342 - 0.355)  # fitted posteriors from NB300 GompertzNBNaive
        g = v["survival_g"]  # 0.019  # fitted posteriors from NB300 GompertzNBNaive

        def _get_e(M, g):
            return np.exp(-g * M)

        surv = stats.gompertz(_get_e(M, g), scale=1 / g).sf(
            x=np.arange(v["age_incept"], v["age_max"] + 1)
        )

        # condition on being alive at age_incept (scale to first pos)
        surv = np.round(surv / surv[0], 4)

        # condition on being alive at age_today (scale to age_today pos)
        idx = v["age_today"] - v["age_incept"]
        df["survival"] = np.minimum(1.0, np.round(surv / surv[idx], 4))[1:]

        # expected losses (death benefit)
        df["e_death_benefit_csum"] = np.round(
            df["death_benefit"] * (1 - df["survival"])
        )

        # expected investment return factor, optionally consider churn
        if v["churn"]:
            # assume P(churn, survive) ~ P(churn|survive) * P(survive) ~ P(churn) * P(survive)
            # basic churn curve reasonable?
            # churn_shapeard = np.geomspace(0.03, 0.00001, len(df))
            # df['churn'] = np.exp(-churn_shapeard.cumsum())
            # basic generalised churn curve
            # NOTE:
            #   + based on a Weibull CDF with shape conditional on age
            #   + same as survival, except for fun use different parameterisation
            #   + Calc from age_incept to allow scaling (based on conditioning below)
            #     trim this before adding to df to start at end-of-year age_incept+1
            churn = stats.weibull_min.sf(
                x=np.arange(v["age_incept"], v["age_max"] + 1),
                c=0.01
                + (v["churn_shape"] / 10)
                * v["age_incept"],  # + churn_shape * v['age_incept'] ** 2,
                loc=0,
                scale=v["churn_scale"],
            )
            # condition on being alive at age_incept (scale to first pos)
            churn = np.round(churn / churn[0], 4)
            # condition on not churned at age_today (scale to age_today pos)
            idx = v["age_today"] - v["age_incept"]
            df["churn"] = np.minimum(1.0, np.round(churn / churn[idx], 4))[1:]

            df["e_investment_factor_instant"] = 1 + (
                v["invest_rate"] * df["survival"] * df["churn"]
            )
        else:
            df["e_investment_factor_instant"] = 1 + (v["invest_rate"] * df["survival"])

        # expected investment return factor weighted by / conditional on
        # staggered premium payment schedule
        mx = np.empty((len(df), v["n_prem_payments"]))
        equal_payment_prop = np.round(1 / v["n_prem_payments"], 4)
        for i in range(v["n_prem_payments"]):
            factors = df["e_investment_factor_instant"].copy()
            factors.iloc[:i] = 1
            mx[:, i] = (factors.cumprod() * equal_payment_prop).values
        df["e_investment_factor_csum"] = np.nan_to_num(mx).sum(axis=1)

        # technical premium required to cover the death benefit
        # NOTE:
        #   + we require investment returns @ incept+db years >= death benefit
        #   multiple payments over n years
        e_investment_factor_reqd = df.iloc[int(v["db_covered_by"])][
            "e_investment_factor_csum"
        ]
        tech_prem = np.round(v["death_benefit"] / e_investment_factor_reqd, -3)
        df["tech_prem"] = np.concatenate(
            (
                np.repeat(
                    np.round(tech_prem / v["n_prem_payments"]), v["n_prem_payments"]
                ),
                np.full(len(df) - v["n_prem_payments"], np.nan),
            )
        )
        df["tech_prem_csum"] = np.nan_to_num(df["tech_prem"]).cumsum()
        df["tech_prem_invested_csum"] = (
            df["tech_prem_csum"] * df["e_investment_factor_csum"]
        )

        # internal cost of business based on tech prem, frontloaded
        cost = -v["cost_ratio"] * tech_prem
        df["cost"] = np.concatenate(
            (
                np.repeat(
                    np.round(cost / v["n_years_frontload"]), v["n_years_frontload"]
                ),
                np.full(len(df) - v["n_years_frontload"], np.nan),
            )
        )
        df["cost_csum"] = np.nan_to_num(df["cost"]).cumsum()

        # internal profit based on tech prem, frontloaded
        profit = v["profit_ratio"] * tech_prem
        df["profit"] = np.concatenate(
            (
                np.repeat(
                    np.round(profit / v["n_years_frontload"]), v["n_years_frontload"]
                ),
                np.full(len(df) - v["n_years_frontload"], np.nan),
            )
        )
        df["profit_csum"] = np.nan_to_num(df["profit"]).cumsum()

        # calc internally-loaded prem
        internal_prem = tech_prem - cost + profit

        # external brokerage paid as prop of the total external (gross) premium
        # (frontloaded on year 0)
        brokerage = -internal_prem * (v["brokerage_ratio"] / (1 - v["brokerage_ratio"]))
        df["brokerage"] = np.concatenate(
            (np.repeat(brokerage, 1), np.full(len(df) - 1, np.nan))
        )
        df["brokerage_csum"] = np.nan_to_num(df["brokerage"]).cumsum()

        # external prem (including brokerage frontloaded on year 0)
        total_prem = internal_prem - brokerage
        df["prem"] = np.concatenate(
            (
                np.repeat(
                    np.round(total_prem / v["n_prem_payments"]), v["n_prem_payments"]
                ),
                np.full(len(df) - v["n_prem_payments"], np.nan),
            )
        )
        df["prem_csum"] = np.nan_to_num(df["prem"]).cumsum()
        df["prem_invested_csum"] = df["prem_csum"] * df["e_investment_factor_csum"]

        # annual rate of return
        df["e_annual_ror"] = (
            df["tech_prem_invested_csum"]
            / (
                np.nan_to_num(df["tech_prem_invested_csum"].shift(1))
                + np.nan_to_num(df["prem"])
            )
        ) - 1

        df["tech_prem_cashval_ng_csum"] = np.maximum(
            np.zeros(len(df)), df["tech_prem_invested_csum"] + df["death_benefit"].max()
        )

        # expected net cashflow for policyholder
        # get annual diffs
        fts = ["prem_csum", "tech_prem_cashval_ng_csum", "e_death_benefit_csum"]
        dfd = df[fts].diff(1).fillna(df[fts])
        df["e_pol_cashflow"] = (
            -dfd["prem_csum"]
            + dfd["tech_prem_cashval_ng_csum"]
            - dfd["e_death_benefit_csum"]  # add e_db to cashflow to policyholder
        )

        # expected net cashflow for primary insurer
        # NOTE ignore 'brokerage', 'cost','profit',
        fts = ["tech_prem_invested_csum", "e_death_benefit_csum"]
        df["e_ins_reserve_csum"] = df[fts].sum(axis=1)

        return df

    def plot_policy_lifetime(self, **kwargs) -> plt.Figure:
        """Convenience plot various financial metrics for a single policy in df
        If df not found in kwargs it will go create it
        If ref_vals dict found in kwargs, it will pass that when creating df,
        otherwise it will create a ref_vals dict based on all kwargs (this is a
        dirty hack to allow wgts.interact to pass in separate vals)
        """
        df = kwargs.pop("df", pd.DataFrame())
        censor100 = kwargs.pop("censor100", False)
        ref_vals = kwargs.pop("ref_vals", dict(**kwargs))

        # NOTE: ridiculous hack to persist the df when created by observe inside
        # wgts.interactive_output. There's no way to capture an outputted df.
        self._df = self.create(ref_vals=ref_vals) if df.empty else df

        # summary values for plotting convenience
        self._df["e_death_benefit_csum_inv"] = -self._df["e_death_benefit_csum"]

        v = [
            "e_death_benefit_csum",
            "prem",
            "cost",
            "brokerage",
            "profit",
            "tech_prem_invested_csum",
            "tech_prem_cashval_ng_csum",
            "e_ins_reserve_csum",
            "e_pol_cashflow",
            "e_investment_factor_csum",
            "e_annual_ror",
            "survival",
        ]

        if "churn" in self._df.columns.values:
            v += ["churn"]

        # censor plot
        if censor100:
            dfp = self._df.loc[self._df["age_of_insured_eoty"] <= 100].copy()
        else:
            dfp = self._df.copy()

        dfm = dfp.reset_index().melt(id_vars="age_of_insured_eoty", value_vars=v)

        dfm["kind"] = dfm["variable"].map(
            {
                "death_benefit": "policy_operations",
                "e_death_benefit_csum": "policy_operations_amt",
                "prem": "policy_operations_amt",
                "cost": "policy_operations_amt",
                "brokerage": "policy_operations_amt",
                "profit": "policy_operations_amt",
                "e_death_benefit_csum_inv": "policy_financials_amt",
                "tech_prem_invested_csum": "policy_financials_amt",
                "tech_prem_cashval_ng_csum": "policy_financials_amt",
                "e_ins_reserve_csum": "policy_financials_amt",
                "e_pol_cashflow": "policy_financials_amt",
                "e_investment_factor_csum": "proportions",
                "e_annual_ror": "proportions",
                "survival": "proportions",
                "churn": "proportions",
            }
        )

        g = sns.relplot(
            x="age_of_insured_eoty",
            y="value",
            hue="variable",
            style="variable",
            col="kind",
            col_wrap=3,
            data=dfm,
            kind="line",
            height=10,
            aspect=0.55,
            estimator=None,
            markers=True,
            lw=1,
            ms=12,
            facet_kws=dict(legend_out=True, sharey=False),
            palette="tab20",
        )
        finminscale = -self._df["death_benefit"].max() // 10
        investscale = np.round(self._df["e_investment_factor_csum"].max() / 10, 2)
        _ = g.axes[0].yaxis.set_major_locator(plt.MultipleLocator(finminscale))
        _ = g.axes[1].yaxis.set_major_locator(plt.MultipleLocator(finminscale * 2))
        _ = g.axes[2].yaxis.set_major_locator(plt.MultipleLocator(investscale))
        # _ = sns.move_legend(g, "lower left", bbox_to_anchor=(0.06, 0.06), frameon=True)
        _ = g.fig.suptitle(
            "Financial metrics for parameterised policy from POV of the Primary Insurer",
            fontsize=18,
        )

        # overplot summary table
        # summary table for incept -> term aka max_age considered (121 default)
        nm = f'ipt ({ref_vals["age_incept"]}) -> T'
        smry0 = (
            self._df.agg(
                premium=pd.NamedAgg(column="prem", aggfunc=np.nansum),
                prem_payments=pd.NamedAgg(column="prem", aggfunc=np.nanmax),
                brokerage=pd.NamedAgg(column="brokerage", aggfunc=np.nansum),
                cost=pd.NamedAgg(column="cost", aggfunc=np.nansum),
                profit=pd.NamedAgg(column="profit", aggfunc=np.nansum),
                db_term=pd.NamedAgg(column="death_benefit", aggfunc=np.max),
                e_cashval_ng_term=pd.NamedAgg(
                    column="tech_prem_cashval_ng_csum", aggfunc=np.max
                ),
                e_annual_ror_max=pd.NamedAgg(column="e_annual_ror", aggfunc=np.max),
            )
            .max(axis=1)
            .to_frame(nm)
        )  # NOTE pandas agg yields 2D df so collapse to series with max(axis=1)

        # append new scalars: compound RoR and conventionally calculated IRR
        # NOTE: for irr this is incept to term, assume original policyholder
        # prem_csum is payment aka deposits (neg by convention)
        # withdrawals (expected values cond. on survival) (pos by convention)
        # be VERY careful about signs when creating net cashflow position
        # use diffs of csum to get incremental payments & withdrawals
        df_smryx0 = pd.DataFrame(
            dict(
                e_compound_ror_term=np.prod(self._df["e_annual_ror"] + 1) - 1,
                e_irr_pol=npf.irr(self._df["e_pol_cashflow"]),
            ),
            index=[nm],
        ).T
        smry0 = pd.concat((smry0, df_smryx0), axis=0)

        # summary table for age_today -> term
        nm = f'tdy ({ref_vals["age_today"]}) -> T'
        idx = self._df["age_of_insured_eoty"] >= (ref_vals["age_today"] + 1)
        smry1 = (
            self._df.loc[idx]
            .agg(
                premium=pd.NamedAgg(column="prem", aggfunc=np.nansum),
                prem_payments=pd.NamedAgg(column="prem", aggfunc=np.nanmax),
                brokerage=pd.NamedAgg(column="brokerage", aggfunc=np.nansum),
                cost=pd.NamedAgg(column="cost", aggfunc=np.nansum),
                profit=pd.NamedAgg(column="profit", aggfunc=np.nansum),
                db_term=pd.NamedAgg(column="death_benefit", aggfunc=np.max),
                e_cashval_ng_term=pd.NamedAgg(
                    column="tech_prem_cashval_ng_csum", aggfunc=np.max
                ),
                e_cashval_ng_eoty=pd.NamedAgg(
                    column="tech_prem_cashval_ng_csum", aggfunc=np.min
                ),  # this is a dummy val to be replaced
                e_db_eoty=pd.NamedAgg(
                    column="e_death_benefit_csum", aggfunc=np.min
                ),  # this is a dummy val to be replaced
                e_annual_ror_max=pd.NamedAgg(column="e_annual_ror", aggfunc=np.max),
            )
            .max(axis=1)
            .to_frame(nm)
        )  # NOTE pandas agg yields 2D df so collapse to series with max(axis=1)

        # get e_cashval_ng_eoty and e_db_eoty
        idx1 = self._df["age_of_insured_eoty"] == (ref_vals["age_today"] + 1)
        smry1.loc["e_cashval_ng_eoty"] = self._df.loc[
            idx1, "tech_prem_cashval_ng_csum"
        ].values
        smry1.loc["e_db_eoty"] = self._df.loc[idx1, "e_death_benefit_csum"].values

        # append new scalars: compound RoR and IRR (based on cashflow)
        # NOTE: for irr this is incept to term, assume original policyholder
        # prem_csum is payment aka deposits (neg by convention)
        # withdrawals (expected values cond. on survival) (pos by convention)
        # be VERY careful about signs when creating net cashflow position
        # use diffs of csum to get incremental payments & withdrawals

        # get cash val as-at today?
        idx_eoty = self._df["age_of_insured_eoty"] == ref_vals["age_today"] + 1
        cashval_eoty = self._df.loc[idx_eoty, "tech_prem_invested_csum"].values
        # if len(cashval_eoty) == 0:
        #     cashval_eoty = [0]

        # create a fair_price today as cashval_eoty # ignore costs + profit
        # and subtract it from the first value of cashflow of the buyer
        fair_price = cashval_eoty[0]  # * (
        #     1 + (ref_vals['cost_ratio'] + ref_vals['profit_ratio'])
        # )
        cashflow = self._df.loc[idx, "e_pol_cashflow"]
        cashflow.iloc[0] = cashflow.iloc[0] - fair_price

        df_smryx1 = pd.DataFrame(
            dict(
                e_compound_ror_term=np.prod(self._df.loc[idx, "e_annual_ror"] + 1) - 1,
                e_irr_pol=npf.irr(cashflow),
            ),
            index=[nm],
        ).T
        smry1 = pd.concat((smry1, df_smryx1), axis=0)

        # create the joined 2-col table for presentation
        smry = pd.merge(smry0, smry1, how="outer", left_index=True, right_index=True)
        smry = smry.reindex(
            [
                "premium",
                "prem_payments",
                "brokerage",
                "cost",
                "profit",
                "db_term",
                "e_cashval_ng_term",
                "e_compound_ror_term",
                "e_annual_ror_max",
                "e_db_eoty",
                "e_cashval_ng_eoty",
                "e_irr_pol",
            ]
        )

        # make signs consistent for ease of reading
        fts = ["e_cashval_ng_term", "e_cashval_ng_eoty"]
        smry.loc[fts] = -smry.loc[fts]

        smry.iloc[:7] = smry.iloc[:7].map("$ {:,.0f}".format)
        smry.iloc[7:9] = smry.iloc[7:9].map("{:,.1%}".format)
        smry.iloc[9:11] = smry.iloc[9:11].map("$ {:,.0f}".format)
        smry.iloc[11:12] = smry.iloc[11:12].map("{:,.1%}".format)

        # hilariously dirty hacks for colours because alpha doesn't work
        kws_tbl = dict(
            cellLoc="right",
            rowLoc="center",
            loc="right",
            zorder=2,
            fontsize=11,
            cellColours=np.ones_like(smry) * ["#ffffff80"],
            rowColours=["#ffffffaa"] * len(smry),
        )
        _ = g.axes[0].table(
            cellText=smry.values,
            rowLabels=smry.index,
            colLabels=smry.columns,
            bbox=[0.64, 0.72, 0.36, 0.28],
            colColours=[
                "#4B93C333",
                "#FF983E33",
            ],  # https://www.color-hex.com/color/1f78b4
            **kws_tbl,
        )
        _ = g.fig.tight_layout()
        return g.fig

    def get_interactive(self) -> tuple[wgts.VBox, wgts.Output, dict]:
        """Convenience, create UI and get interactive output
        BE CAREFUL OF DOUBLE PLOTTING RESULTING FROM
        https://github.com/jupyter-widgets/ipywidgets/blob/e0d41f6f02324596a282bc9e4650fd7ba63c0004/ipywidgets/widgets/interaction.py#L26
        DO NOT USE %matplotlib inline in the Notebook!!!
        """

        kws = dict(style=dict(description_width="initial"), continuous_update=True)
        kws1 = dict(layout=wgts.Layout(width="100px"))
        age_incept = wgts.IntSlider(
            value=40, min=20, max=80, step=5, description="Age Incept", **kws
        )
        age_today = wgts.IntSlider(
            value=40, min=20, max=100, step=5, description="Age Today", **kws
        )
        # link controls
        # https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#linking-widgets  # noqa
        _ = wgts.dlink((age_incept, "value"), (age_today, "min"))
        # _ = wgts.dlink((age_incept, 'value'), (age_today, 'value'))  # track?
        # survival_shape = wgts.FloatSlider(
        #     value=0.09,
        #     min=0.05,
        #     max=0.18,
        #     step=0.01,
        #     description='Survival Shape',
        #     **kws,
        # )
        # survival_scale = wgts.IntSlider(
        #     value=70, min=60, max=80, step=2, description='Survival Scale', **kws
        # )
        M = np.round(np.exp(4.342), 1)  # fitted posteriors from NB300 GompertzNBNaive
        g = 0.019  # fitted posteriors from NB300 GompertzNBNaive
        survival_m = wgts.FloatSlider(
            value=M,
            min=0.1 * M,
            max=2 * M,
            step=0.1 * M,
            description="Survival M",
            readout_format=".0f",
            **kws,
        )
        survival_g = wgts.FloatSlider(
            value=g,
            min=0.1 * g,
            max=2 * g,
            step=0.1 * g,
            description="Survival Gamma",
            **kws,
        )
        death_benefit = wgts.IntSlider(
            value=500000,
            min=250000,
            max=2000000,
            step=250000,
            description="Death Ben ($)",
            **kws,
        )
        db_covered_by = wgts.IntSlider(
            value=20, min=5, max=40, step=5, description="DB Covered @", **kws
        )
        n_prem_payments = wgts.IntSlider(
            value=10, min=1, max=20, step=1, description="n Prem Payments", **kws
        )
        invest_rate = wgts.FloatSlider(
            value=0.04, min=0, max=0.1, step=0.01, description="Inv. Rate", **kws
        )
        brokerage = wgts.FloatSlider(
            value=0.2, min=0.0, max=0.3, step=0.02, description="Brokerage", **kws
        )
        cost = wgts.FloatSlider(
            value=0.1, min=0.0, max=0.2, step=0.02, description="Internal Costs", **kws
        )
        profit = wgts.FloatSlider(
            value=0.1, min=0.0, max=0.2, step=0.02, description="Target Profit", **kws
        )
        censor100 = wgts.Checkbox(value=True, description="Censor plot age 100", **kws)
        churn = wgts.Checkbox(value=False, description="Consider Churn", **kws)
        churn_shape = wgts.FloatSlider(
            value=0.1, min=0.0, max=0.2, step=0.05, description="Churn Shape", **kws
        )
        churn_scale = wgts.IntSlider(
            value=30, min=20, max=40, step=2, description="Churn Scale", **kws
        )
        controls = dict(
            age_incept=age_incept,
            age_today=age_today,
            # survival_shape=survival_shape,
            # survival_scale=survival_scale,
            survival_m=survival_m,
            survival_g=survival_g,
            death_benefit=death_benefit,
            db_covered_by=db_covered_by,
            n_prem_payments=n_prem_payments,
            invest_rate=invest_rate,
            brokerage_ratio=brokerage,
            cost_ratio=cost,
            profit_ratio=profit,
            censor100=censor100,
            churn=churn,
            churn_shape=churn_shape,
            churn_scale=churn_scale,
        )
        ui = wgts.VBox(
            [
                wgts.Label("Whole Life Policy Simulation Parameters"),
                wgts.HBox(
                    [
                        wgts.Label("Policyholder:", **kws1),
                        age_incept,
                        age_today,
                        # survival_shape,
                        # survival_scale,
                        survival_m,
                        survival_g,
                    ]
                ),
                wgts.HBox(
                    [
                        wgts.Label("Policy:", **kws1, min_width="80px"),
                        death_benefit,
                        db_covered_by,
                        n_prem_payments,
                    ]
                ),
                wgts.HBox(
                    [
                        wgts.Label("Operations:", **kws1, min_width="80px"),
                        invest_rate,
                        brokerage,
                        cost,
                        profit,
                    ]
                ),
                wgts.HBox(
                    [
                        wgts.Label("Misc:", **kws1, min_width="80px"),
                        censor100,
                        churn,
                        churn_shape,
                        churn_scale,
                    ]
                ),
            ]
        )
        out = wgts.interactive_output(self.plot_policy_lifetime, controls)
        return ui, out, controls
