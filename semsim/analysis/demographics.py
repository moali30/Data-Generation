"""
semsim.analysis.demographics
============================
What each demographic role produces once the data exist.

Every variable gets a frequency table setting the requested share against the
achieved one, and every pair of variables the researcher linked gets a
chi-square test of independence with Cramér's V.  A variable set aside for *comparison* gets a test of the
difference on every construct's composite score: Welch's t-test for two groups
and a one-way ANOVA for three or more.  A variable set aside for *multi-group
analysis* gets a sequence of nested fits across its groups: configural, equal
loadings (metric invariance) and equal loadings plus equal paths (structural
invariance).

Scalar invariance is not offered.  It tests intercepts, and the estimator here
fits covariance structures with no mean structure, so reporting it would mean
inventing a number.

The multi-group fits use the substantive model only.  A control variable
describes who the respondents are, and including it would change what is being
compared across groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize as _opt
from scipy import stats as _stats

from semsim.analysis import estimator as _estimator
from semsim.core.compile import compile_model
from semsim.core.errors import SemSimError
from semsim.core.spec import (
    Construct, DemographicRole, DemographicVariable, EstimatorKind, ModelSpec,
    ScaleKind,
)

__all__ = ["DemographicAnalysis", "analyse", "frequency_table",
           "association_tests", "compare_groups", "invariance_tests"]

ALPHA = 0.05

#: Cheung & Rensvold's (2002) cut-off for a negligible drop in CFI.
DELTA_CFI_LIMIT = 0.01

#: Parameters held equal at each invariance step, by estimator kind.
_METRIC = (_estimator.KIND_LOADING, _estimator.KIND_SECOND_ORDER)
_STRUCTURAL = _METRIC + (_estimator.KIND_PATH,)

FREQUENCY_COLUMNS = ["variable", "category_ar", "category_en", "role",
                     "target_percent", "n", "achieved_percent", "difference"]
ASSOCIATION_COLUMNS = ["variable_a", "variable_b", "chi_square", "df", "p",
                       "cramers_v", "significant"]
COMPARISON_COLUMNS = ["variable", "construct", "test", "statistic", "df1", "df2",
                      "p", "effect_size", "effect_kind", "means_ar", "means_en",
                      "significant"]
INVARIANCE_COLUMNS = ["variable", "model_ar", "model_en", "chi_square", "df", "p",
                      "cfi", "rmsea", "delta_chi_square", "delta_df", "delta_p",
                      "delta_cfi", "invariant"]
GROUP_PATH_COLUMNS = ["variable", "group_ar", "group_en", "n", "source", "target",
                      "std", "p"]


@dataclass
class DemographicAnalysis:
    """The output of :func:`analyse`.

    Attributes
    ----------
    frequencies : pandas.DataFrame
        One row per category of every variable.
    associations : pandas.DataFrame
        One row per linked pair of variables.
    comparisons : pandas.DataFrame
        One row per comparison variable and construct.
    invariance : pandas.DataFrame
        One row per multi-group variable and nested model.
    group_paths : pandas.DataFrame
        Each group's standardised paths from its configural fit.
    notes : list of str
        Bilingual remarks, such as a group too small to fit.
    """

    frequencies: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=FREQUENCY_COLUMNS))
    associations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=ASSOCIATION_COLUMNS))
    comparisons: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=COMPARISON_COLUMNS))
    invariance: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=INVARIANCE_COLUMNS))
    group_paths: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=GROUP_PATH_COLUMNS))
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Frequencies
# --------------------------------------------------------------------------- #


def frequency_table(spec: ModelSpec, demographics: pd.DataFrame) -> pd.DataFrame:
    """Requested and achieved share of every category.

    Parameters
    ----------
    spec : ModelSpec
        Supplies the variables, their categories and their target shares.
    demographics : pandas.DataFrame
        The dataset's demographic columns.

    Returns
    -------
    pandas.DataFrame
        Columns in :data:`FREQUENCY_COLUMNS`, percentages on a 0-100 scale.

    References
    ----------
    Moser, C. A. (1952). Quota sampling. *Journal of the Royal Statistical
        Society, Series A*, 115(3), 411-423.
    """
    rows: List[Dict[str, Any]] = []
    for variable in spec.demographics:
        if variable.name not in demographics.columns:
            continue
        column = demographics[variable.name].to_numpy()
        total = column.size or 1
        for code, (category, share) in enumerate(
                zip(variable.categories, variable.normalised_proportions()),
                start=1):
            count = int(np.count_nonzero(column == code))
            rows.append({
                "variable": variable.name,
                "category_ar": category.label.ar or category.label.en,
                "category_en": category.label.en or category.label.ar,
                "role": variable.role.value,
                "target_percent": share * 100.0,
                "n": count,
                "achieved_percent": count / total * 100.0,
                "difference": count / total * 100.0 - share * 100.0,
            })
    return pd.DataFrame(rows, columns=FREQUENCY_COLUMNS)


# --------------------------------------------------------------------------- #
# Associations
# --------------------------------------------------------------------------- #


def association_tests(spec: ModelSpec, demographics: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of every pair of demographic variables the model links.

    Parameters
    ----------
    spec : ModelSpec
        Supplies the variables and their associations.
    demographics : pandas.DataFrame
        The dataset's demographic columns.

    Returns
    -------
    pandas.DataFrame
        Columns in :data:`ASSOCIATION_COLUMNS`: Pearson's chi-square without
        continuity correction, its degrees of freedom and p value, and
        Cramér's V.

    References
    ----------
    Cramér, H. (1946). *Mathematical Methods of Statistics*, section 21.9.
        Princeton University Press.
    Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.), chapter 3.
        Wiley.
    """
    from semsim.core.associations import chi_square, valid_associations
    variables = {v.id: v for v in spec.demographics}
    rows: List[Dict[str, Any]] = []
    for association in valid_associations(spec):
        first, second = variables[association.first], variables[association.second]
        if (first.name not in demographics.columns
                or second.name not in demographics.columns):
            continue
        test = chi_square(demographics[first.name].to_numpy(),
                          demographics[second.name].to_numpy(),
                          len(first.categories), len(second.categories))
        if test is None:
            continue
        statistic, df, p, v = test
        rows.append({"variable_a": first.name, "variable_b": second.name,
                     "chi_square": statistic, "df": df, "p": p,
                     "cramers_v": v, "significant": bool(p < ALPHA)})
    return pd.DataFrame(rows, columns=ASSOCIATION_COLUMNS)


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #


def _composite(construct: Construct, data: pd.DataFrame) -> Optional[np.ndarray]:
    """Row mean of a construct's items, reverse-worded items recoded first."""
    columns: List[np.ndarray] = []
    for item in construct.all_items():
        if item.name not in data.columns:
            continue
        values = data[item.name].to_numpy(dtype=float)
        if item.reverse_coded:
            values = (item.scale.n_categories + 1 - values
                      if item.scale.kind is ScaleKind.LIKERT else -values)
        columns.append(values)
    if not columns:
        return None
    matrix = np.column_stack(columns)
    filled = np.sum(np.isfinite(matrix), axis=1)
    total = np.nansum(matrix, axis=1)
    out = np.full(matrix.shape[0], np.nan)
    out[filled > 0] = total[filled > 0] / filled[filled > 0]
    return out


def _welch(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float]:
    """Welch's t, its degrees of freedom, the p value and Cohen's d."""
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    na, nb = a.size, b.size
    se2 = va / na + vb / nb
    t = (float(a.mean()) - float(b.mean())) / np.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = float(2.0 * _stats.t.sf(abs(t), df))
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float(t), float(df), p, (float(a.mean()) - float(b.mean())) / float(pooled)


def _anova(groups: Sequence[np.ndarray]) -> Tuple[float, int, int, float, float]:
    """One-way F, both degrees of freedom, the p value and eta squared."""
    pooled = np.concatenate(groups)
    grand = float(pooled.mean())
    between = sum(g.size * (float(g.mean()) - grand) ** 2 for g in groups)
    within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    df1, df2 = len(groups) - 1, pooled.size - len(groups)
    f = (between / df1) / (within / df2)
    return (float(f), df1, df2, float(_stats.f.sf(f, df1, df2)),
            float(between / (between + within)))


def compare_groups(spec: ModelSpec, data: pd.DataFrame,
                   demographics: pd.DataFrame,
                   notes: Optional[List[str]] = None) -> pd.DataFrame:
    """Test every comparison variable's differences on every construct.

    Parameters
    ----------
    spec : ModelSpec
        Supplies the comparison variables and the constructs.
    data : pandas.DataFrame
        The item responses.
    demographics : pandas.DataFrame
        The dataset's demographic columns, in the same row order.
    notes : list of str, optional
        Receives a bilingual remark for each comparison that cannot be run.

    Returns
    -------
    pandas.DataFrame
        Columns in :data:`COMPARISON_COLUMNS`.  ``effect_size`` is Cohen's d
        for two groups and eta squared for three or more.

    References
    ----------
    Welch, B. L. (1947). The generalization of "Student's" problem when several
        different population variances are involved. *Biometrika*, 34(1-2),
        28-35.
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
        (2nd ed.). Erlbaum.
    """
    rows: List[Dict[str, Any]] = []
    sink = notes if notes is not None else []
    for variable in spec.demographics:
        if (variable.role is not DemographicRole.COMPARISON
                or variable.name not in demographics.columns):
            continue
        codes = demographics[variable.name].to_numpy()
        for construct in spec.constructs:
            score = _composite(construct, data)
            if score is None:
                continue
            groups, kept = [], []
            for code, category in enumerate(variable.categories, start=1):
                values = score[(codes == code) & np.isfinite(score)]
                if values.size >= 2 and float(values.var(ddof=1)) > 1e-12:
                    groups.append(values)
                    kept.append(category)
            if len(groups) < 2:
                sink.append(
                    f"{variable.name}: too few cases with varying scores to "
                    f"compare groups on {construct.name}. | {variable.name}: "
                    f"لا توجد حالات كافية متباينة للمقارنة على {construct.name}.")
                continue
            if len(groups) == 2:
                statistic, df2, p, effect = _welch(groups[0], groups[1])
                test, df1, kind = "welch_t", np.nan, "cohen_d"
            else:
                statistic, df1, df2, p, effect = _anova(groups)
                test, kind = "anova_f", "eta_squared"
            rows.append({
                "variable": variable.name, "construct": construct.name,
                "test": test, "statistic": statistic, "df1": df1, "df2": df2,
                "p": p, "effect_size": effect, "effect_kind": kind,
                "means_ar": " | ".join(
                    f"{c.label.ar or c.label.en}: {g.mean():.2f}"
                    for c, g in zip(kept, groups)),
                "means_en": " | ".join(
                    f"{c.label.en or c.label.ar}: {g.mean():.2f}"
                    for c, g in zip(kept, groups)),
                "significant": bool(p < ALPHA),
            })
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


# --------------------------------------------------------------------------- #
# Multi-group invariance
# --------------------------------------------------------------------------- #


@dataclass
class _Group:
    """One group's sample moments and its configural fit."""

    category: Any
    n: int
    sample_cov: np.ndarray
    logdet: float
    weight: float
    theta: np.ndarray


class _MultiGroupProblem:
    """The summed maximum-likelihood discrepancy with some parameters shared.

    The vector is the shared parameters followed by each group's own, so a
    change to one group's own parameter only re-evaluates that group.
    """

    def __init__(self, par: Any, groups: List[_Group],
                 shared_kinds: Sequence[str]) -> None:
        self.par = par
        self.groups = groups
        self.shared = [k for k, prm in enumerate(par.parameters)
                       if prm.kind in shared_kinds]
        shared = set(self.shared)
        self.own = [k for k in range(par.q) if k not in shared]
        self.n_shared = len(self.shared)
        self.n_own = len(self.own)

    def start(self) -> np.ndarray:
        weights = np.array([g.weight for g in self.groups], dtype=float)
        weights = weights / weights.sum()
        matrix = np.vstack([g.theta for g in self.groups])
        common = (weights @ matrix)[self.shared]
        return np.concatenate([common] + [g.theta[self.own] for g in self.groups])

    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        base = self.par.bounds()
        out = [base[k] for k in self.shared]
        for _ in self.groups:
            out.extend(base[k] for k in self.own)
        return out

    def _group_theta(self, x: np.ndarray, g: int) -> np.ndarray:
        theta = np.zeros(self.par.q)
        theta[self.shared] = x[: self.n_shared]
        start = self.n_shared + g * self.n_own
        theta[self.own] = x[start: start + self.n_own]
        return theta

    def _f(self, theta: np.ndarray, group: _Group) -> float:
        sigma = self.par.sigma(theta)
        if sigma is None:
            return float("inf")
        return group.weight * _estimator.ml_discrepancy(
            group.sample_cov, sigma, logdet_sample=group.logdet)

    def chi_square(self, x: np.ndarray) -> float:
        return float(sum(self._f(self._group_theta(x, g), group)
                         for g, group in enumerate(self.groups)))

    def value_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        step = 1e-6
        total = 0.0
        gradient = np.zeros(x.size)
        scale = max(sum(g.weight for g in self.groups), 1.0)
        for g, group in enumerate(self.groups):
            theta = self._group_theta(x, g)
            base = self._f(theta, group)
            if not np.isfinite(base):
                return float("inf"), gradient
            total += base
            offset = self.n_shared + g * self.n_own
            for position, k in enumerate(self.shared + self.own):
                nudged = theta.copy()
                delta = step * max(abs(theta[k]), 1.0)
                nudged[k] += delta
                bumped = self._f(nudged, group)
                if not np.isfinite(bumped):
                    nudged[k] = theta[k] - delta
                    bumped = 2.0 * base - self._f(nudged, group)
                slope = (bumped - base) / delta
                if position < self.n_shared:
                    gradient[position] += slope
                else:
                    gradient[offset + position - self.n_shared] += slope
        return total / scale, gradient / scale

    def solve(self) -> Tuple[np.ndarray, bool]:
        with _estimator._quiet():
            result = _opt.minimize(
                self.value_and_gradient, self.start(), jac=True,
                method="L-BFGS-B", bounds=self.bounds(),
                options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-6})
        return np.asarray(result.x, dtype=float), bool(result.success)


def _baseline_chi_square(groups: Sequence[_Group]) -> float:
    """Chi-square of the independence model, summed over groups."""
    total = 0.0
    for group in groups:
        diagonal = np.clip(np.diag(group.sample_cov), 1e-12, None)
        total += group.weight * (float(np.sum(np.log(diagonal))) - group.logdet)
    return total


def _cfi(chi2: float, df: int, chi2_null: float, df_null: int) -> float:
    numerator = max(chi2 - df, 0.0)
    denominator = max(chi2_null - df_null, numerator, 1e-12)
    return 1.0 - numerator / denominator


def _rmsea(chi2: float, df: int, n_total: int, n_groups: int) -> float:
    """Multi-group RMSEA (Steiger, 1998)."""
    if df <= 0:
        return 0.0
    return float(np.sqrt(max(chi2 - df, 0.0) / (df * max(n_total - n_groups, 1)))
                 * np.sqrt(n_groups))


def _model_row(variable: DemographicVariable, ar: str, en: str, chi2: float,
               df: int, chi2_null: float, df_null: int, n_total: int,
               n_groups: int, previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variable": variable.name, "model_ar": ar, "model_en": en,
        "chi_square": chi2, "df": df,
        "p": float(_stats.chi2.sf(chi2, df)) if df > 0 else np.nan,
        "cfi": _cfi(chi2, df, chi2_null, df_null),
        "rmsea": _rmsea(chi2, df, n_total, n_groups),
        "delta_chi_square": np.nan, "delta_df": np.nan, "delta_p": np.nan,
        "delta_cfi": np.nan, "invariant": None,
    }
    if previous is not None:
        delta = max(chi2 - previous["chi_square"], 0.0)
        delta_df = df - previous["df"]
        delta_p = float(_stats.chi2.sf(delta, delta_df)) if delta_df > 0 else np.nan
        delta_cfi = previous["cfi"] - row["cfi"]
        row.update(delta_chi_square=delta, delta_df=delta_df, delta_p=delta_p,
                   delta_cfi=delta_cfi,
                   invariant=bool(delta_cfi <= DELTA_CFI_LIMIT
                                  or (delta_p == delta_p and delta_p >= ALPHA)))
    return row


def invariance_tests(spec: ModelSpec, data: pd.DataFrame,
                     demographics: pd.DataFrame,
                     notes: Optional[List[str]] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the configural, metric and structural models across groups.

    Parameters
    ----------
    spec : ModelSpec
        Supplies the multi-group variables and the model.
    data : pandas.DataFrame
        The item responses.
    demographics : pandas.DataFrame
        The dataset's demographic columns, in the same row order.
    notes : list of str, optional
        Receives a bilingual remark for each variable that cannot be fitted.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        The nested-model comparison (:data:`INVARIANCE_COLUMNS`) and each
        group's standardised paths (:data:`GROUP_PATH_COLUMNS`).

    References
    ----------
    Joreskog, K. G. (1971). Simultaneous factor analysis in several
        populations. *Psychometrika*, 36(4), 409-426.
    Cheung, G. W., & Rensvold, R. B. (2002). Evaluating goodness-of-fit indexes
        for testing measurement invariance. *Structural Equation Modeling*,
        9(2), 233-255.
    Steiger, J. H. (1998). A note on multiple sample extensions of the RMSEA
        fit index. *Structural Equation Modeling*, 5(4), 411-419.
    """
    sink = notes if notes is not None else []
    variables = [v for v in spec.demographics
                 if v.role is DemographicRole.MGA and v.name in demographics.columns]
    fit_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    if not variables:
        return (pd.DataFrame(columns=INVARIANCE_COLUMNS),
                pd.DataFrame(columns=GROUP_PATH_COLUMNS))

    base = replace(spec, demographics=[])
    cm = compile_model(base)
    notes_local: List[str] = []
    par = _estimator._build_parameterisation(cm, notes_local)
    moments = par.p * (par.p + 1) // 2

    for variable in variables:
        codes = demographics[variable.name].to_numpy()
        groups: List[_Group] = []
        estimates: List[Tuple[Any, Any]] = []
        failed = ""
        for code, category in enumerate(variable.categories, start=1):
            frame = data.loc[codes == code]
            if frame.dropna().shape[0] <= max(par.p + 1, 10):
                failed = category.label.get("en")
                break
            try:
                fit = _estimator.estimate(frame, base, estimator=EstimatorKind.ML,
                                          use_polychoric=False, start_from=cm)
            except SemSimError:
                failed = category.label.get("en")
                break
            logdet, _chol = _estimator._logdet_cholesky(fit.sample_cov)
            groups.append(_Group(
                category=category, n=int(fit.n), sample_cov=fit.sample_cov,
                logdet=float(logdet), weight=float(fit.n - 1),
                theta=np.array([fit.parameters[label] for label in par.labels],
                               dtype=float)))
            estimates.append((category, fit))
        if failed or len(groups) < 2:
            sink.append(
                f"{variable.name}: the group '{failed}' is too small to fit the "
                f"model on its own (it needs more than {par.p + 1} complete "
                f"cases), so invariance was not tested. | {variable.name}: "
                f"المجموعة «{failed}» أصغر من أن يُقدَّر النموذج عليها وحدها "
                f"(تحتاج أكثر من {par.p + 1} حالة مكتملة)، فلم يُختبر الثبات.")
            continue

        k = len(groups)
        n_total = sum(g.n for g in groups)
        chi2_null = _baseline_chi_square(groups)
        df_null = k * par.p * (par.p - 1) // 2
        df_config = k * (moments - par.q)

        free_problem = _MultiGroupProblem(par, groups, ())
        configural = free_problem.chi_square(free_problem.start())
        previous = _model_row(variable, "الشكلي (Configural)", "Configural",
                              configural, df_config, chi2_null, df_null, n_total,
                              k, None)
        fit_rows.append(previous)

        steps = (("ثبات التشبعات (Metric)", "Metric (equal loadings)", _METRIC),
                 ("ثبات المسارات (Structural)", "Structural (equal paths)",
                  _STRUCTURAL))
        converged = True
        for ar, en, kinds in steps:
            problem = _MultiGroupProblem(par, groups, kinds)
            if not problem.shared:
                continue
            x, ok = problem.solve()
            converged = converged and ok
            df = df_config + (k - 1) * problem.n_shared
            previous = _model_row(variable, ar, en, problem.chi_square(x), df,
                                  chi2_null, df_null, n_total, k, previous)
            fit_rows.append(previous)
        if not converged:
            sink.append(
                f"{variable.name}: one of the constrained fits did not fully "
                f"converge; read its chi-square with caution. | {variable.name}: "
                "أحد النماذج المقيَّدة لم يصل إلى التقارب الكامل؛ "
                "فاقرأ قيمة مربع كاي له بحذر.")

        for category, fit in estimates:
            table = fit.path_estimates
            for _index, row in table.iterrows():
                path_rows.append({
                    "variable": variable.name,
                    "group_ar": category.label.ar or category.label.en,
                    "group_en": category.label.en or category.label.ar,
                    "n": int(fit.n),
                    "source": row["source"], "target": row["target"],
                    "std": float(row["std"]), "p": float(row["p"]),
                })
    return (pd.DataFrame(fit_rows, columns=INVARIANCE_COLUMNS),
            pd.DataFrame(path_rows, columns=GROUP_PATH_COLUMNS))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def analyse(spec: ModelSpec, dataset: Any) -> Optional[DemographicAnalysis]:
    """Run every analysis the demographic roles call for.

    Parameters
    ----------
    spec : ModelSpec
        The model as specified.
    dataset : GeneratedDataset
        A generated dataset carrying demographic columns.

    Returns
    -------
    DemographicAnalysis or None
        ``None`` when the dataset has no demographics.

    References
    ----------
    Vandenberg, R. J., & Lance, C. E. (2000). A review and synthesis of the
        measurement invariance literature. *Organizational Research Methods*,
        3(1), 4-70.
    """
    demographics = getattr(dataset, "demographics", None)
    if demographics is None or demographics.empty:
        return None
    out = DemographicAnalysis(frequencies=frequency_table(spec, demographics))
    out.associations = association_tests(spec, demographics)
    out.comparisons = compare_groups(spec, dataset.data, demographics, out.notes)
    out.invariance, out.group_paths = invariance_tests(
        spec, dataset.data, demographics, out.notes)
    return out
