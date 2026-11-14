# SEM Data Simulation Engine — Architecture & Module Contracts

> **This file is the integration contract.** Every module is written against the
> signatures below. Do not change a published signature; add to it.

Stack: **Python 3.11+ / PySide6 (Qt) / numpy / scipy / pandas**. Target: Windows
desktop (PyInstaller one-file). UI is bilingual Arabic (RTL) + English.

## Layering

```
semsim/core/spec.py          ModelSpec dataclasses  (DONE — the contract)
        core/compile.py      ModelSpec -> LISREL matrices, fully standardised
        core/implied.py      Sigma(theta), Phi, total/indirect effects, PD repair
        core/validation.py   bilingual diagnostics (identification, Heywood, cycles)
        core/associations.py Gaussian copula linking demographic variables

semsim/generation/fleishman.py     Fleishman (1978) polynomial coefficients
           generation/vale_maurelli.py  Vale & Maurelli (1983) non-normal MVN
           generation/thresholds.py     continuous y* -> Likert categories
           generation/noise.py          careless responders, outliers, missingness
           generation/careless.py       careless respondents added on top of the sample
           generation/heterogeneity.py  one realistic population around the spec
           generation/group_differences.py  comparison groups that differ as planned
           generation/respondent_styles.py  each respondent's use of the scale
           generation/engine.py         orchestrator: spec -> DataFrame

semsim/targeting/cudeck_browne.py  inject model error for a target RMSEA
           targeting/significance.py   coefficient/N policy for target p-values
           targeting/calibration.py    closed loop: generate -> fit -> adjust

semsim/analysis/estimator.py     ML / ULS / DWLS SEM estimation
          analysis/fit_indices.py  chi2, RMSEA, CFI, TLI, SRMR, GFI, AIC/BIC...
          analysis/reliability.py  alpha, omega, CR, AVE, Fornell-Larcker, HTMT
          analysis/polychoric.py   polychoric / polyserial correlations
          analysis/descriptives.py means, SD, skew, kurtosis, normality tests

semsim/power/analysis.py   MacCallum close-fit power, Satorra-Saris, heuristics

semsim/io/project.py    .semsim JSON round-trip (dataclass <-> dict)
       io/exporters.py  CSV / XLSX / SPSS .sav / lavaan / Mplus syntax
       io/report.py     self-contained bilingual HTML report

semsim/ui/...  PySide6 wizard: 6 pages mirroring the 6 methodology stages
```

Hard rules
* Only `semsim.ui.*` may import PySide6. Everything else is headless and testable.
* Every public function gets a type-annotated signature and a numpydoc-style
  docstring with a `References` section naming the paper it implements.
* All randomness flows through an explicit `numpy.random.Generator`. No global
  `np.random` calls, ever — reproducibility is a product feature.
* Raise `SemSimError` (from `semsim.core.errors`) for user-facing failures;
  never let a bare LinAlgError reach the UI.

---

## `semsim/core/errors.py`

```python
class SemSimError(Exception):
    """Base class; carries a bilingual message."""
    def __init__(self, message_en: str, message_ar: str = "", **context): ...
    @property
    def message_en(self) -> str: ...
    @property
    def message_ar(self) -> str: ...

class SpecificationError(SemSimError): ...
class IdentificationError(SemSimError): ...
class HeywoodCaseError(SemSimError): ...
class ConvergenceError(SemSimError): ...
class InfeasibleTargetError(SemSimError): ...
```

---

## `semsim/core/compile.py`

Flattens the hierarchical `ModelSpec` into one latent vector and the four
LISREL-style matrices, in the **completely standardised** metric: every latent
and every observed item has unit total variance.

```python
@dataclass
class CompiledModel:
    spec: ModelSpec
    item_ids: list[str]          # length p, dataset column order
    item_names: list[str]        # length p
    factor_ids: list[str]        # length m: constructs, dimensions, method factor
    factor_names: list[str]
    Lambda: np.ndarray           # (p, m) standardised loadings
    B: np.ndarray                # (m, m) B[i, j] = effect OF j ON i
    Psi: np.ndarray              # (m, m) latent disturbance covariance
    Theta: np.ndarray            # (p, p) item residual covariance
    Phi: np.ndarray              # (m, m) implied latent covariance (unit diagonal)
    endogenous: np.ndarray       # (m,) bool
    item_index: dict[str, int]
    factor_index: dict[str, int]
    method_factor_index: int | None
    home_factor: list[int]       # length p, index of each item's primary factor
    warnings: list[str]          # Heywood clamping, infeasible targets, ...

def compile_model(spec: ModelSpec, rng: np.random.Generator | None = None) -> CompiledModel
```

### The standardisation algebra (implement exactly this)

Latent model: `eta = B eta + zeta`, so `Phi = Cov(eta) = (I-B)^-1 Psi (I-B)^-T`.
Measurement model: `y = Lambda eta + eps`, `Theta = Cov(eps)`.
Implied covariance: **`Sigma = Lambda Phi Lambda' + Theta`**.

1. **Build the latent node list.** One node per construct, one per dimension,
   plus one method factor when `spec.common_method_bias.enabled`. Order the
   nodes topologically with respect to `B` so recursive models give a
   lower-triangular `B`.
2. **Populate `B`.**
   * second-order loading of dimension `d` on construct `c` -> `B[d, c]`
     (the dimension is *regressed on* the construct);
   * each `Path(source=s, target=t)` -> `B[t, s] = path.resolved_coefficient()`.
3. **Exogenous latents.** `Psi[i, i] = 1`. For each `LatentCovariance(a, b, r)`
   set `Psi[a, b] = Psi[b, a] = r` (both must be exogenous; if either is
   endogenous treat it as a correlated disturbance and warn).
4. **Endogenous latents, in topological order.** For node `i` with predictor
   indices `P` and coefficient row `b = B[i, P]`:

   ```
   explained  = b @ Phi[P][:, P] @ b
   Psi[i, i]  = 1 - explained          # disturbance variance
   ```

   `explained >= 1` is a latent Heywood case: clamp `Psi[i, i]` to
   `HEYWOOD_FLOOR = 1e-3`, rescale `b` by `sqrt((1 - HEYWOOD_FLOOR)/explained)`,
   write the shrunken coefficients back into `B`, and append a warning naming
   the construct and the original/retained coefficients. Then fill in
   `Phi[i, j] = b @ Phi[P, j]` for every already-processed `j`, and
   `Phi[i, i] = 1`.
   For a non-recursive (cyclic) model fall back on fixed-point iteration of
   `Phi = (I-B)^-1 Psi (I-B)^-T` with the diagonal of `Psi` rescaled each pass
   until `diag(Phi)` is within `1e-10` of one (max 500 passes, else
   `ConvergenceError`).
5. **`Lambda`.** Primary loading of each item on its home factor
   (`spec.home_factor_of`), plus every `CrossLoading`, plus the method-factor
   column. Auto-generated cross-loadings come from
   `spec.cross_loading_policy` using the supplied `rng`.
6. **`Theta` diagonal.** `communality_j = Lambda[j] @ Phi @ Lambda[j]`;
   `Theta[j, j] = 1 - communality_j`. If that is `<= 0`, scale the whole row of
   `Lambda` by `sqrt((1 - HEYWOOD_FLOOR)/communality_j)`, set
   `Theta[j, j] = HEYWOOD_FLOOR`, and warn.
7. **Unreliability.** Apply `item.extra_error_variance` and
   `spec.noise.unique_variance_inflation` by *inflating* `Theta[j, j]` and then
   renormalising the row to unit total variance:
   `total = communality_j + Theta_inflated`; divide `Lambda[j]` by
   `sqrt(total)` and `Theta[j, j]` by `total`. This attenuates the standardised
   loading, which is exactly what weaker measurement does.
8. **Error covariances.** `Theta[a, b] = Theta[b, a] = r * sqrt(Theta[a,a] * Theta[b,b])`.
   Guard `|r| < 1`. After filling, if `Theta` is not positive definite, repair it
   with `implied.nearest_positive_definite` and warn.

---

## `semsim/core/implied.py`

```python
def latent_covariance(cm: CompiledModel) -> np.ndarray            # Phi (m, m)
def implied_covariance(cm: CompiledModel) -> np.ndarray           # Sigma (p, p)
def implied_correlation(cm: CompiledModel) -> np.ndarray
def total_effects(cm: CompiledModel) -> np.ndarray                # (I-B)^-1 - I
def direct_effect(cm, source_id: str, target_id: str) -> float
def indirect_effect(cm, chain: Sequence[str]) -> float            # product along the chain
def all_indirect_effects(cm, source_id, target_id, max_len: int = 4) -> list[tuple[list[str], float]]
def population_reliability(cm) -> dict[str, dict[str, float]]
    # per factor id: {"ave", "cr", "alpha_implied", "omega", "n_items"}
def population_htmt(cm) -> tuple[np.ndarray, list[str]]
def is_positive_definite(m: np.ndarray, tol: float = 1e-10) -> bool
def nearest_positive_definite(m: np.ndarray, eps: float = 1e-8) -> np.ndarray
    # Higham (2002) alternating projections; must preserve symmetry
def condition_number(m: np.ndarray) -> float
```

---

## `semsim/core/validation.py`

```python
class Severity(str, Enum): ERROR = "error"; WARNING = "warning"; INFO = "info"

@dataclass
class Diagnostic:
    severity: Severity
    code: str                    # stable machine code, e.g. "DF_NEGATIVE"
    message_en: str
    message_ar: str
    where: str = ""              # construct / item / path id
    hint_en: str = ""
    hint_ar: str = ""

@dataclass
class ValidationReport:
    diagnostics: list[Diagnostic]
    @property
    def ok(self) -> bool                # no ERROR entries
    def errors(self) -> list[Diagnostic]
    def warnings(self) -> list[Diagnostic]

def validate(spec: ModelSpec) -> ValidationReport
def validate_compiled(cm: CompiledModel) -> ValidationReport
```

Checks to implement (codes in brackets):
`NO_CONSTRUCTS`, `CONSTRUCT_NO_ITEMS`, `TOO_FEW_ITEMS` (<3 on a factor is
under-identified in isolation), `SECOND_ORDER_TOO_FEW_DIMS` (<3 dimensions
needs extra constraints), `DUPLICATE_ITEM_NAME`, `LOADING_OUT_OF_RANGE`
(|lambda| >= 1), `NEGATIVE_DF`, `ZERO_DF` (just-identified, fit indices are
degenerate), `CYCLE_IN_PATHS`, `SELF_PATH`, `DUPLICATE_PATH`,
`PATH_UNKNOWN_CONSTRUCT`, `MODERATOR_NOT_IN_INTERACTION`,
`INTERACTION_TARGET_EXOGENOUS`, `MEDIATOR_WITHOUT_IN_OR_OUT`,
`DEPENDENT_WITHOUT_PREDICTOR`, `ERRCOV_SAME_ITEM`, `ERRCOV_OUT_OF_RANGE`,
`CROSSLOAD_HOME_FACTOR`, `HEYWOOD_RISK` (communality >= 1),
`R2_EXCEEDS_ONE`, `LIKERT_CATEGORY_RANGE` (outside 2..11),
`PROPORTIONS_DONT_SUM`, `INFEASIBLE_SKEW_KURTOSIS`
(kurtosis <= skew^2 - 2, Fleishman has no real solution),
`SAMPLE_SIZE_TOO_SMALL` (N < 5 * free parameters),
`FIT_TARGET_IMPOSSIBLE_AT_ZERO_DF`, `MISSING_PROPORTION_RANGE`.

Every diagnostic needs a *useful* Arabic message, not a transliteration.

---

## `semsim/generation/fleishman.py`

```python
FLEISHMAN_LOWER_KURTOSIS_BOUND = lambda skew: skew**2 - 2

@dataclass
class FleishmanCoefficients:
    a: float; b: float; c: float; d: float
    skewness: float; kurtosis: float
    converged: bool; iterations: int; residual: float

def is_feasible(skewness: float, kurtosis: float) -> bool
def solve(skewness: float, kurtosis: float, *, tol=1e-10, max_iter=500) -> FleishmanCoefficients
    # Newton-Raphson on Fleishman's three equations; a = -c.
    # skew == 0 and kurtosis == 0 must return exactly (0, 1, 0, 0).
def transform(z: np.ndarray, coef: FleishmanCoefficients) -> np.ndarray
    # Y = a + b z + c z^2 + d z^3
def intermediate_correlation(target_rho: float, ci: FleishmanCoefficients,
                             cj: FleishmanCoefficients) -> float
    # Solve Vale & Maurelli's cubic in rho_z:
    #   rho_y = rho_z (b_i b_j + 3 b_i d_j + 3 d_i b_j + 9 d_i d_j)
    #         + rho_z^2 (2 c_i c_j) + rho_z^3 (6 d_i d_j)
    # Pick the real root in (-1, 1) closest to target_rho; raise
    # InfeasibleTargetError when none exists.
```

References: Fleishman (1978) *Psychometrika* 43, 521-532;
Vale & Maurelli (1983) *Psychometrika* 48, 465-471.

## `semsim/generation/vale_maurelli.py`

```python
def generate(n: int, sigma: np.ndarray, skewness: Sequence[float],
             kurtosis: Sequence[float], rng: np.random.Generator,
             *, repair_pd: bool = True) -> np.ndarray     # (n, p)
    # 1. Fleishman coefficients per variable.
    # 2. Intermediate *correlation* matrix pairwise; repair to PD if needed.
    # 3. Cholesky -> correlated standard normals -> polynomial transform.
    # 4. Rescale to the SDs implied by diag(sigma) and add the means.
def generate_normal(n, sigma, rng, mean=None) -> np.ndarray
def achieved_moments(x: np.ndarray) -> dict[str, np.ndarray]   # mean/sd/skew/excess kurtosis
```

## `semsim/generation/thresholds.py`

```python
def thresholds_for_style(n_categories: int, style: ResponseStyle) -> np.ndarray
    # z-cut points, length n_categories - 1, strictly increasing.
    # SYMMETRIC: equal-width cuts of Beta(3.5, 3.5) -- a centred bell, not flat
    # marginals. UNIFORM is the equal-probability one: cuts at Phi^-1(k/K).
    # POSITIVE_SKEW/NEGATIVE_SKEW: beta-shaped proportions (a, b) = (2, 5)/(5, 2);
    # EXTREME_*: (1.4, 6)/(6, 1.4); BIMODAL: mixture with mass at both ends.
def thresholds_from_proportions(proportions: Sequence[float]) -> np.ndarray
def categorise(y_star: np.ndarray, thresholds: np.ndarray, *, base: int = 1) -> np.ndarray
def apply_scale(y_star: np.ndarray, scale: ScaleSpec) -> np.ndarray
    # Likert -> integers in [1, n_categories]; continuous -> mean + sd * y_star.
def reverse_code(values: np.ndarray, scale: ScaleSpec) -> np.ndarray
def achieved_proportions(values: np.ndarray, n_categories: int) -> np.ndarray
def polychoric_attenuation(thresholds: np.ndarray) -> float
    # How much a Pearson correlation shrinks under this categorisation; used by
    # the calibration loop to pre-compensate loadings.
```

## `semsim/generation/noise.py`

```python
@dataclass
class NoiseReport:
    n_careless: int; careless_rows: list[int]
    n_outlier_cells: int; n_missing_cells: int
    applied: list[str]

def inject_careless(data: np.ndarray, scales: list[ScaleSpec], proportion: float,
                    style: str, rng) -> tuple[np.ndarray, list[int]]
    # "realistic" (default): careless_patterns() on the chosen rows.
    # "random": uniform draws over the response scale.
    # "straightline": one category repeated across the whole row.
    # "mixed": half and half.
def careless_patterns(block, scales, rng, *, reference=None) -> tuple[np.ndarray, list[str]]
    # Each row becomes a straight-liner (35%), a late straight-liner who goes
    # flat from 40-80% of the way down (35%) or a random responder (30%).
    # The repeated code is the midpoint for a third of them and otherwise
    # drawn from the attentive sample's pooled answers; a random answer comes
    # half from the item's own distribution, half uniform over the scale.
def inject_outliers(data, proportion, magnitude_sd, rng, scales=None) -> tuple[np.ndarray, int]
def inject_missing(data, mechanism, proportion, rng, driver=None) -> tuple[np.ndarray, int]
    # MCAR: uniform. MAR: logistic in the driver column, so missingness depends
    # on an observed score (returns float array with np.nan).
def apply_noise(data: np.ndarray, spec: ModelSpec, cm: CompiledModel,
                rng, *, careless=True) -> tuple[np.ndarray, NoiseReport]
    # careless=False skips the careless step (the engine passes it when
    # spec.noise.careless_screened adds them on top instead).
```

## `semsim/generation/careless.py`

```python
@dataclass
class CarelessBlock:
    items: pd.DataFrame; demographics: pd.DataFrame | None
    positions: list[int]          # zero-based rows of the file, ascending
    kinds: list[str]              # straightline / partial / random
    file_rows -> list[int]        # one-based, as a spreadsheet shows them

def extra_count(spec, n) -> int
    # k with k / (n + k) = noise.careless_proportion, when careless_screened.
def draw_careless(spec, y_star, data, demographics, scales, score, k, seed)
    # From its own stream [seed, 49979687]: first answers drawn from the
    # attentive latent responses' mean and covariance and scored like theirs,
    # made careless by careless_patterns(); demographics copied from random
    # attentive rows; positions scattered over the n + k rows.
def insert_rows(frame, extra, positions) -> pd.DataFrame
```

`GeneratedDataset.data` and `.demographics` stay the analysed sample (every
target is calibrated on it). `.careless` holds the block; `.full_frame` is the
file with the careless rows in place, `.clean_frame` the sample without them,
`.n_in_file` the file's row count and `.in_file_order(frame)` lines a
per-respondent extra (latent scores, continuous shadow) up with the file.

## `semsim/generation/heterogeneity.py`

```python
def realise_population(spec: ModelSpec, seed: int) -> ModelSpec
    # A deep copy describing one population around the specification, drawn
    # from its own stream keyed on the run seed (spec.heterogeneity):
    #   - each Likert item gets explicit cut points, shifted by its construct's
    #     level plus its own difficulty and stretched by its own spread;
    #   - loadings and second-order loadings move around the planned value;
    #   - paths meant to be significant, and exogenous correlations, move but
    #     keep their sign and at least 75% of their size.
    #   - tail_share of each item's answers is spread over all categories;
    #   - reverse-worded items get mirrored cut points and a loading
    #     2 * loading_sd weaker (the engine mirrors style-shaped reverse items
    #     itself when variation is off).
    # Explicit thresholds/proportions, null paths, interactions and anything
    # touching a control term are left alone. A change that compiles with more
    # warnings than the original is undone. The copy has heterogeneity
    # switched off, so it is never varied twice. calibrate() calls it once per
    # run; generate_dataset() calls it when given no compiled model.
```

## `semsim/core/associations.py`

```python
def latent_structure(spec: ModelSpec) -> LatentStructure
    # Every associated demographic variable gets a standard normal latent
    # score; its categories are slices at its shares, in list order. Each
    # link's latent correlation is found by bisection so the bivariate
    # normal cross-table has the requested Cramér's V (sign from the
    # direction). Unlinked pairs get the maximum-determinant completion
    # (a chain A-B-C implies r_AC = r_AB * r_BC); a set that cannot hold
    # is softened to the nearest positive definite matrix, with a note.
def term_correlation(...) -> float
    # Population correlation of two control terms (codes or dummies) of
    # linked variables; ModelSpec.expanded() writes it into the model.
def chi_square(codes_a, codes_b, k, m) -> (chi2, df, p, V) | None
```

`generation/demographics.draw_demographics_jointly` draws every variable as
before, then replaces the linked ones with slices of correlated latent
scores (exact quotas fill by rank). With significance targeting on, the
scores are redrawn until each link's chi-square test matches its flag.
The scores travel to `apply_group_differences`, where a linked comparison
variable's propensity includes `E[latent | linked latents]`, so regrouping
keeps the association.

## `semsim/generation/group_differences.py`

```python
def apply_group_differences(spec: ModelSpec, data: pd.DataFrame,
                            demographics: pd.DataFrame | None, seed: int
                            ) -> tuple[pd.DataFrame | None, list[str]]
    # For every comparison variable with group_differences, regroup the
    # respondents by a propensity u = Z w + sqrt(1 - rho' w) e, where Z are
    # the target constructs' standardised composites (the comparison
    # table's own scores), rho the correlations that give each target its
    # eta squared (d converted for two groups) once u is cut at the group
    # shares, and w = R^-1 rho. The group sizes already drawn are filled in
    # order of u, so the answers, the counts and every SEM result are
    # untouched. With significance targeting on, e is redrawn (up to
    # MAX_ATTEMPTS) until each test is significant or not as asked; the
    # closest draw is kept otherwise, with a warning. generate_dataset()
    # calls it once the answers are final.
```

## `semsim/generation/respondent_styles.py`

```python
def draw_respondent_styles(spec, n, seed) -> RespondentStyles | None
    # From its own stream (spec.respondent_styles): a stretch
    # s = exp(sd_e z - sd_e^2 / 2) (extreme versus midpoint) and a shift
    # a = sd_a z (acquiescence) per respondent. _apply_scales maps each
    # rating item's latent response y to m + s (y - m) + a before cutting,
    # where m is the middle of the item's cut points. Reverse items carry
    # a negative loading, so the shift agrees with them as written.
    # The continuous shadow, the demographics and every other stream are
    # untouched; continuous items are left alone.
```

## `semsim/generation/engine.py`

```python
@dataclass
class GeneratedDataset:
    data: pd.DataFrame            # observed items, dataset column order
    latent_scores: pd.DataFrame | None
    continuous_shadow: pd.DataFrame | None   # y* before categorisation
    compiled: CompiledModel
    sigma_population: np.ndarray  # Sigma(theta) with no model error
    sigma_generating: np.ndarray  # after Cudeck-Browne injection
    n: int
    seed: int
    noise_report: NoiseReport | None
    engine_used: GenerationEngine
    warnings: list[str]

def generate_dataset(spec: ModelSpec, *, n: int | None = None,
                     seed: int | None = None,
                     progress: Callable[[str, float], None] | None = None
                     ) -> GeneratedDataset
```

---

## `semsim/analysis/estimator.py`

```python
@dataclass
class EstimationResult:
    converged: bool; n_iter: int; estimator: EstimatorKind
    n: int; n_free: int; df: int
    fmin: float                     # discrepancy at the optimum
    sigma_hat: np.ndarray           # model-implied, at theta_hat
    sample_cov: np.ndarray
    parameters: dict[str, float]            # labelled estimates
    standard_errors: dict[str, float]
    z_values: dict[str, float]
    p_values: dict[str, float]
    standardised: dict[str, float]
    loadings: pd.DataFrame          # item, factor, est, se, z, p, std
    path_estimates: pd.DataFrame    # source, target, est, se, z, p, std
    r_squared: dict[str, float]     # per endogenous construct
    message: str

def estimate(data: pd.DataFrame, spec: ModelSpec, *,
             estimator: EstimatorKind = EstimatorKind.ML,
             use_polychoric: bool | None = None,
             start_from: CompiledModel | None = None) -> EstimationResult
```

Discrepancy functions
* ML: `F = log|Sigma| + tr(S Sigma^-1) - log|S| - p`
* ULS: `F = 0.5 * tr((S - Sigma)^2)`
* DWLS: `F = (s - sigma)' diag(W)^-1 (s - sigma)` on the polychoric matrix

Optimise with `scipy.optimize.minimize` (L-BFGS-B, then Nelder-Mead as a
fallback), starting from the compiled population values when available and
otherwise from a two-step factor-analytic start. Standard errors come from the
inverse expected information `2/(N-1) * (Delta' W Delta)^-1` with `Delta` the
numerical Jacobian of `vech(Sigma)` and `W = 0.5 * D'(Sigma^-1 ⊗ Sigma^-1) D`.
Parameters are bounded to keep variances positive; a non-PD `Sigma` during the
search returns `+inf` rather than raising.

## `semsim/analysis/fit_indices.py`

```python
@dataclass
class FitIndices:
    chi_square: float; df: int; p_value: float; chi_square_df_ratio: float
    rmsea: float; rmsea_ci_lower: float; rmsea_ci_upper: float; rmsea_pclose: float
    cfi: float; tli: float; nfi: float; ifi: float; rfi: float
    srmr: float; gfi: float; agfi: float
    aic: float; bic: float; sabic: float
    baseline_chi_square: float; baseline_df: int
    n: int; n_free: int
    def verdict(self) -> FitLevel
    def to_table(self, lang: str = "en") -> pd.DataFrame
        # index, value, threshold, pass/fail, bilingual interpretation

def compute(sample_cov, sigma_hat, n, n_free, *, n_items=None) -> FitIndices
def baseline_fit(sample_cov, n) -> tuple[float, int]       # independence model
def rmsea_confidence_interval(chi2, df, n, level=0.90) -> tuple[float, float]
    # invert the non-central chi-square; scipy.optimize.brentq on ncx2.cdf
def thresholds_table(lang="en") -> pd.DataFrame            # Hu & Bentler (1999)
```

## `semsim/analysis/reliability.py`

```python
@dataclass
class FactorReliability:
    factor_id: str; factor_name: str; n_items: int
    cronbach_alpha: float; alpha_ci: tuple[float, float]
    composite_reliability: float        # Joreskog's rho_c
    omega_total: float
    ave: float
    sqrt_ave: float
    mean_inter_item_r: float
    item_total_correlations: dict[str, float]
    alpha_if_item_dropped: dict[str, float]

@dataclass
class ValidityReport:
    per_factor: list[FactorReliability]
    factor_correlations: pd.DataFrame
    fornell_larcker: pd.DataFrame        # sqrt(AVE) on the diagonal
    fornell_larcker_passed: bool
    htmt: pd.DataFrame
    htmt_passed: bool
    htmt_threshold: float
    cross_loading_table: pd.DataFrame
    flags: list[str]

def cronbach_alpha(x: np.ndarray) -> float
def composite_reliability(loadings, error_variances) -> float
def average_variance_extracted(loadings) -> float
def htmt_matrix(data, factor_items: dict[str, list[str]]) -> pd.DataFrame
def analyse(data: pd.DataFrame, spec: ModelSpec,
            estimation: EstimationResult | None = None,
            *, use_polychoric: bool = False,
            htmt_threshold: float = 0.85) -> ValidityReport
```

`composite_reliability = (sum lambda)^2 / ((sum lambda)^2 + sum theta)`,
`AVE = mean(lambda^2)`. Prefer estimated loadings when `estimation` is given,
otherwise fall back on a per-factor principal-axis solution.

## `semsim/analysis/polychoric.py`

```python
def polychoric_correlation(x: np.ndarray, y: np.ndarray, *,
                           max_categories: int = 11) -> float
    # Two-step: thresholds from the marginals, then 1-D ML on rho via the
    # bivariate normal CDF (scipy.stats.multivariate_normal / Drezner-Wesolowsky).
def polyserial_correlation(ordinal: np.ndarray, continuous: np.ndarray) -> float
def polychoric_matrix(data: pd.DataFrame, ordinal_columns=None,
                      *, repair_pd: bool = True) -> pd.DataFrame
def bivariate_normal_cdf(h: float, k: float, rho: float) -> float
```

## `semsim/analysis/descriptives.py`

```python
@dataclass
class ItemDescriptives:
    name: str; n: int; n_missing: int
    mean: float; sd: float; median: float; minimum: float; maximum: float
    skewness: float; kurtosis: float              # excess, Fisher
    se_skewness: float; se_kurtosis: float
    z_skewness: float; z_kurtosis: float
    shapiro_w: float | None; shapiro_p: float | None
    category_proportions: dict[int, float] | None
    floor_pct: float; ceiling_pct: float

@dataclass
class DescriptivesReport:
    items: list[ItemDescriptives]
    mardia_skewness: float; mardia_skewness_p: float
    mardia_kurtosis: float; mardia_kurtosis_p: float
    multivariate_normal: bool
    n: int
    def to_table(self, lang="en") -> pd.DataFrame

def describe(data: pd.DataFrame, spec: ModelSpec | None = None) -> DescriptivesReport
def mardia_test(x: np.ndarray) -> tuple[float, float, float, float]
```

---

## `semsim/power/analysis.py`

```python
@dataclass
class PowerResult:
    method: PowerMethod
    n_recommended: int
    power_at_recommended: float
    alpha: float; df: int; n_free: int
    curve: pd.DataFrame                  # n, power — for plotting
    rule_of_thumb: dict[str, int]        # {"n_per_parameter", "n_per_item", "minimum_200", ...}
    per_path_power: pd.DataFrame | None  # source, target, coefficient, n_required, power
    notes_en: list[str]; notes_ar: list[str]

def power_close_fit(df: int, n: int, *, rmsea_null=0.05, rmsea_alt=0.08,
                    alpha=0.05) -> float
    # MacCallum, Browne & Sugawara (1996). Non-central chi-square:
    # crit = chi2.ppf(1-alpha, df, loc=0, scale=1) under lambda_null,
    # power = 1 - ncx2.cdf(crit, df, lambda_alt), lambda = (n-1) * df * rmsea^2.
def n_for_close_fit_power(df, target_power=0.80, **kw) -> int      # bisection
def power_not_close_fit(df, n, *, rmsea_null=0.05, rmsea_alt=0.01, alpha=0.05) -> float
def satorra_saris_power(spec, cm, path: Path, n: int, alpha=0.05) -> float
    # Fit the restricted model (path fixed to 0) to the population Sigma;
    # noncentrality = (n-1) * F_restricted, df = 1.
def n_for_path_power(spec, cm, path, target_power=0.80, alpha=0.05) -> int
def monte_carlo_power(spec, *, n, replications=200, alpha=0.05, seed=None,
                      progress=None) -> pd.DataFrame
def analyse(spec: ModelSpec, cm: CompiledModel | None = None) -> PowerResult
```

---

## `semsim/targeting/cudeck_browne.py`

```python
@dataclass
class ModelErrorResult:
    sigma_0: np.ndarray            # the generating matrix, PD
    achieved_rmsea: float
    achieved_fmin: float
    target_rmsea: float
    scale: float                   # how far along the perturbation direction
    converged: bool
    warnings: list[str]

def inject_model_error(cm: CompiledModel, target_rmsea: float, *,
                       df: int | None = None, rng=None,
                       method: str = "cudeck_browne",
                       tol: float = 1e-8, max_iter: int = 200) -> ModelErrorResult
```

Cudeck & Browne (1992): choose `Sigma_0 = Sigma(theta) + k E` such that
`F_ML(Sigma_0, Sigma(theta)) = F_0 = df * target_rmsea^2` **and `theta` remains
the exact minimiser**. Achieve the second condition by projecting a random
symmetric direction out of the model's tangent space in the ML metric:

```
Delta = d vech(Sigma) / d theta                      # numerical Jacobian
W     = 0.5 * D' (Sigma^-1 ⊗ Sigma^-1) D             # ML weight, D = duplication
E     = unvech( (I - Delta (Delta' W Delta)^-1 Delta' W) vech(E_random) )
```

Then solve `F_ML(Sigma + k E, Sigma) = F_0` for the scalar `k` by Brent's
method, shrinking the bracket whenever `Sigma + k E` loses positive
definiteness. `target_rmsea <= 0` or `df <= 0` returns `Sigma(theta)` unchanged.
The `"residual"` method skips the projection (cheaper, slightly biased).

References: Cudeck & Browne (1992) *Psychometrika* 57, 357-369;
Browne & Cudeck (1993) in *Testing Structural Equation Models*.

## `semsim/targeting/significance.py`

```python
@dataclass
class SignificancePlan:
    coefficients: dict[str, float]       # path id -> coefficient to generate with
    n_required: int
    per_path: pd.DataFrame               # path id, label, target, coefficient, expected power
    notes: list[str]

def build_plan(spec: ModelSpec, cm: CompiledModel, n: int) -> SignificancePlan
    # Paths flagged non-significant get spec.significance_target.null_coefficient.
    # Paths flagged significant keep their coefficient; n_required is the max over
    # paths of n_for_path_power(...).
def check_realised(estimation: EstimationResult, spec: ModelSpec,
                   alpha: float = 0.05) -> tuple[bool, pd.DataFrame]
    # Did every targeted path land on the right side of alpha?
```

## `semsim/targeting/calibration.py`

```python
@dataclass
class CalibrationResult:
    dataset: GeneratedDataset
    estimation: EstimationResult
    fit: FitIndices
    rounds: int
    draws: int
    populations: int             # simulated populations tried (1 to 3)
    history: pd.DataFrame        # round, population, rmsea_target, rmsea_realised, adjustment
    significance_ok: bool
    significance_table: pd.DataFrame
    converged: bool
    warnings: list[str]

def calibrate(spec: ModelSpec, *, n=None, seed=None, progress=None) -> CalibrationResult
    # Closed loop: generate -> estimate -> compare realised RMSEA and path
    # p-values against the targets -> adjust the injected F_0 by the ratio
    # (target/realised)^2 and re-draw. Honour
    # spec.fit_target.max_calibration_rounds and .tolerance, plus
    # spec.significance_target.max_resamples for the p-value side.
    # All draws of one pass sample one population realised around the plan
    # (heterogeneity). When that population misses a requested p-value,
    # up to EXTRA_POPULATIONS (2) further populations are tried, seeded
    # apart so the first pass is unchanged. Draws are scored by
    # miss_distance(): missed paths, then log10 distance of p from alpha;
    # the closest draw is kept when none matches.
```

---

## Demographics and control variables

`ModelSpec.demographics` holds `DemographicVariable`s. Each has categories with
proportions and one role: control, descriptive, group comparison or multi-group.

- `semsim/generation/demographics.py` assigns categories from its own random
  stream (`default_rng([abs(seed), 104729])`), so adding a descriptive variable
  never changes the questionnaire items. `ModelSpec.quota_exact` selects exact
  quotas (largest remainder) or weighted random sampling.
- `ModelSpec.expanded()` turns each active control term into a synthetic
  single-indicator `Construct(covariate=True)`, appended after the substantive
  constructs, with `Path`s that keep the `ControlPath` ids and free covariances
  among the terms and with the exogenous constructs. It returns `self` when no
  control is active, so models without controls generate identical numbers.
- The compiler excludes covariates from Heywood handling, reliability
  inflation, automatic cross-loadings and common method variance. Noise and
  response scales act on questionnaire items only.
- A nominal control is dummy-coded with its first category as the reference.
  Within one variable the terms correlate at
  `-sqrt(p_i p_j / ((1 - p_i) (1 - p_j)))`.
- `Dataset.data` is items only. `Dataset.demographics` holds codes and dummies,
  `Dataset.model_frame` is control terms plus items for estimation, and
  `Dataset.full_frame` is every demographic plus items for export and preview.
- The estimator starts control terms on their coded metric
  (`_rescale_covariate_starts`). With control terms present, an inadmissible
  point returns a finite penalty rather than `+inf`, and a run that never left
  the start is not accepted as converged.
- `semsim/analysis/demographics.py` produces the frequency table, Welch t-tests
  and one-way ANOVA, and configural, metric and structural invariance tests.
  Scalar invariance is not offered because the model has no mean structure.

## `semsim/io/ai_template.py`

A JSON document an AI assistant fills from a plain description of a study.
`build_template()` holds the instructions, the allowed values and a valid
worked example. It uses construct names rather than internal ids, so an
assistant never has to invent or match ids.

`from_text()` accepts an answer wrapped in chat text and collects every problem
before raising `AiTemplateError`, each with its location and both languages.
After the structural read it runs `validate()`, so a template is held to the
same rules as a model built by hand. Changing a key means updating `_TOP_KEYS`
or the matching `reader.keys(...)` call, the instructions and the example.

## `semsim/io/project.py`

```python
PROJECT_SUFFIX = ".semsim"
def to_dict(spec: ModelSpec) -> dict
def from_dict(payload: dict) -> ModelSpec        # tolerant of older spec_version
def save(spec: ModelSpec, path: str | os.PathLike) -> str   # the path actually written
def load(path: str | os.PathLike) -> ModelSpec
def describe_project(path) -> dict               # cheap header read for a recents list
```

Round-tripping must be exact: enums by value, tuples restored as tuples,
`Bilingual` as `{"ar": ..., "en": ...}`. Unknown keys are ignored with a warning
rather than raising, so newer files degrade gracefully.

## `semsim/io/exporters.py`

```python
def export_csv(ds: GeneratedDataset, path, *, decimal_places=4) -> None
def export_xlsx(ds, path, *, include_reports=None) -> None    # data + metadata sheets
def export_sav(ds, path, spec) -> None                        # pyreadstat, with labels
def export_rds_script(ds, path) -> None
def lavaan_syntax(spec: ModelSpec) -> str
def mplus_syntax(spec: ModelSpec, data_filename: str = "data.dat") -> str
def amos_notes(spec: ModelSpec, lang="en") -> str
def export_all(ds, directory, spec, reports=None) -> list[str]
```

`lavaan_syntax` must emit `=~` measurement lines (second-order constructs get
their own `=~` on the dimension names), `~` regressions, `~~` residual
covariances, and `:=` definitions for every requested indirect effect.

## `semsim/io/report.py`

```python
@dataclass
class ReportBundle:
    spec: ModelSpec
    dataset: GeneratedDataset
    estimation: EstimationResult | None
    fit: FitIndices | None
    validity: ValidityReport | None
    descriptives: DescriptivesReport | None
    power: PowerResult | None
    calibration: CalibrationResult | None

def render_html(bundle: ReportBundle, *, lang="ar", embed_charts=True) -> str
def write_html(bundle, path, **kw) -> None
def render_markdown(bundle, *, lang="ar") -> str
```

One self-contained HTML file: inline CSS, `dir="rtl"` for Arabic, charts as
inline SVG (no external requests, no CDN). Sections mirror the six methodology
stages, each table carrying the conventional threshold next to the obtained
value and a bilingual verdict.
