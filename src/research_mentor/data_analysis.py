"""Statistical analysis pipeline for tabular artifacts.

When a student uploads CSV/TSV data, the LLM selects appropriate statistical
tests from a fixed palette, then scipy/pandas execute them deterministically.

Pipeline:
1. pandas basics (already in text_extraction._extract_csv)
2. LLM reads metadata → outputs AnalysisPlan (structured_call)
3. Execute plan deterministically (scipy/pandas)
4. Format results as human-readable text
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.llm import structured_call

# ---------------------------------------------------------------------------
# Schemas (LLM structured output)
# ---------------------------------------------------------------------------

TEST_NAMES = Literal[
    "shapiro", "t_test", "one_way_anova", "tukey_hsd",
    "chi_square", "pearson_correlation", "mann_whitney",
    "linear_regression", "logistic_regression", "two_way_anova",
    "pca",
]


class Analysis(BaseModel):
    """A single statistical test selected by the LLM."""

    test: TEST_NAMES
    columns: list[Annotated[str, Field(max_length=150)]] = Field(max_length=10)
    grouping_column: str | None = Field(default=None, max_length=150)
    reasoning: str = Field(max_length=500)


class AnalysisPlan(BaseModel):
    """LLM-selected set of tests, or a reason to skip."""

    analyses: list[Analysis] = Field(max_length=10)
    skip_reason: str | None = Field(default=None, max_length=500)


class AnalysisResult(BaseModel):
    """Result of executing a single statistical test."""

    test: str
    columns: list[str]
    grouping_column: str | None = None
    reasoning: str
    result: str
    warning: str | None = None


# ---------------------------------------------------------------------------
# Minimum sample sizes per test
# ---------------------------------------------------------------------------

_MIN_SAMPLES: dict[str, int] = {
    "shapiro": 3,
    "t_test": 3,       # at least 2 per group, but 3 total minimum
    "one_way_anova": 3,
    "tukey_hsd": 3,
    "chi_square": 5,
    "pearson_correlation": 3,
    "mann_whitney": 3,
    "linear_regression": 5,     # need more observations than predictors
    "logistic_regression": 10,  # convergence needs reasonable sample
    "two_way_anova": 6,         # at least 2 groups × 2 factors × 1+ obs
    "pca": 5,                   # need more rows than columns
}


# ---------------------------------------------------------------------------
# scipy test execution
# ---------------------------------------------------------------------------


def _run_test(analysis: Analysis, df: pd.DataFrame) -> AnalysisResult:
    """Execute a single statistical test on the dataframe.

    Returns an AnalysisResult with the formatted result string,
    or a warning if the test couldn't run.
    """
    from scipy import stats

    base = AnalysisResult(
        test=analysis.test,
        columns=analysis.columns,
        grouping_column=analysis.grouping_column,
        reasoning=analysis.reasoning,
        result="",
    )

    # Validate columns exist
    missing = [c for c in analysis.columns if c not in df.columns]
    if analysis.grouping_column and analysis.grouping_column not in df.columns:
        missing.append(analysis.grouping_column)
    if missing:
        base.warning = f"Column(s) not found: {', '.join(missing)}"
        base.result = "Skipped"
        return base

    try:
        match analysis.test:
            case "shapiro":
                col = analysis.columns[0]
                data = df[col].dropna()
                if len(data) < _MIN_SAMPLES["shapiro"]:
                    base.warning = f"Too few observations ({len(data)}) for Shapiro-Wilk"
                    base.result = "Skipped"
                    return base
                stat, p = stats.shapiro(data)
                normal = "normal" if p > 0.05 else "not normal"
                base.result = f"Shapiro-Wilk ({col}): W={stat:.4f}, p={p:.4f} → {normal} (α=0.05)"

            case "t_test":
                col = analysis.columns[0]
                grp = analysis.grouping_column
                if grp is None:
                    base.warning = "t-test requires a grouping column"
                    base.result = "Skipped"
                    return base
                groups = _get_groups(df, col, grp)
                if isinstance(groups, str):
                    base.warning = groups
                    base.result = "Skipped"
                    return base
                if len(groups) != 2:
                    base.warning = f"t-test requires exactly 2 groups, found {len(groups)}"
                    base.result = "Skipped"
                    return base
                g1, g2 = groups
                if len(g1) < 2 or len(g2) < 2:
                    base.warning = "Each group needs at least 2 observations"
                    base.result = "Skipped"
                    return base
                stat, p = stats.ttest_ind(g1, g2)
                n1, n2 = len(g1), len(g2)
                deg_f = n1 + n2 - 2
                base.result = (
                    f"Independent t-test ({col} by {grp}): "
                    f"t({deg_f})={stat:.3f}, p={p:.4f}"
                )

            case "one_way_anova":
                col = analysis.columns[0]
                grp = analysis.grouping_column
                if grp is None:
                    base.warning = "ANOVA requires a grouping column"
                    base.result = "Skipped"
                    return base
                groups = _get_groups(df, col, grp)
                if isinstance(groups, str):
                    base.warning = groups
                    base.result = "Skipped"
                    return base
                if len(groups) < 2:
                    base.warning = f"ANOVA requires at least 2 groups, found {len(groups)}"
                    base.result = "Skipped"
                    return base
                for g in groups:
                    if len(g) < 2:
                        base.warning = "Each group needs at least 2 observations"
                        base.result = "Skipped"
                        return base
                stat, p = stats.f_oneway(*groups)
                k = len(groups)
                n = sum(len(g) for g in groups)
                base.result = (
                    f"One-way ANOVA ({col} by {grp}): "
                    f"F({k - 1},{n - k})={stat:.3f}, p={p:.4f}"
                )

            case "tukey_hsd":
                col = analysis.columns[0]
                grp = analysis.grouping_column
                if grp is None:
                    base.warning = "Tukey HSD requires a grouping column"
                    base.result = "Skipped"
                    return base
                groups = _get_groups(df, col, grp)
                if isinstance(groups, str):
                    base.warning = groups
                    base.result = "Skipped"
                    return base
                if len(groups) < 2:
                    base.warning = f"Tukey HSD requires at least 2 groups, found {len(groups)}"
                    base.result = "Skipped"
                    return base
                for g in groups:
                    if len(g) < 2:
                        base.warning = "Each group needs at least 2 observations"
                        base.result = "Skipped"
                        return base
                group_labels = df[grp].dropna().unique()
                result = stats.tukey_hsd(*groups)
                lines = [f"Tukey HSD ({col} by {grp}):"]
                for i in range(len(group_labels)):
                    for j in range(i + 1, len(group_labels)):
                        p_val = result.pvalue[i][j]
                        sig = "*" if p_val < 0.05 else "ns"
                        lines.append(
                            f"  {group_labels[i]} vs {group_labels[j]}: p={p_val:.4f} {sig}"
                        )
                base.result = "\n".join(lines)

            case "chi_square":
                if len(analysis.columns) < 2:
                    base.warning = "Chi-square requires 2 columns"
                    base.result = "Skipped"
                    return base
                col1, col2 = analysis.columns[0], analysis.columns[1]
                contingency = pd.crosstab(df[col1], df[col2])
                if contingency.size < 4:
                    base.warning = "Contingency table too small for chi-square"
                    base.result = "Skipped"
                    return base
                chi2, p, dof, _ = stats.chi2_contingency(contingency)  # type: ignore[attr-defined]
                base.result = (
                    f"Chi-square ({col1} × {col2}): "
                    f"χ²({dof})={chi2:.3f}, p={p:.4f}"
                )

            case "pearson_correlation":
                if len(analysis.columns) < 2:
                    base.warning = "Pearson correlation requires 2 columns"
                    base.result = "Skipped"
                    return base
                col1, col2 = analysis.columns[0], analysis.columns[1]
                valid = df[[col1, col2]].dropna()
                if len(valid) < _MIN_SAMPLES["pearson_correlation"]:
                    base.warning = f"Too few observations ({len(valid)}) for Pearson"
                    base.result = "Skipped"
                    return base
                if valid[col1].std() == 0 or valid[col2].std() == 0:
                    base.warning = "Constant column — correlation undefined"
                    base.result = "Skipped"
                    return base
                r, p = stats.pearsonr(valid[col1], valid[col2])
                base.result = f"Pearson correlation ({col1} ↔ {col2}): r={r:.4f}, p={p:.4f}"

            case "mann_whitney":
                col = analysis.columns[0]
                grp = analysis.grouping_column
                if grp is None:
                    base.warning = "Mann-Whitney requires a grouping column"
                    base.result = "Skipped"
                    return base
                groups = _get_groups(df, col, grp)
                if isinstance(groups, str):
                    base.warning = groups
                    base.result = "Skipped"
                    return base
                if len(groups) != 2:
                    base.warning = f"Mann-Whitney requires exactly 2 groups, found {len(groups)}"
                    base.result = "Skipped"
                    return base
                g1, g2 = groups
                if len(g1) < 1 or len(g2) < 1:
                    base.warning = "Each group needs at least 1 observation"
                    base.result = "Skipped"
                    return base
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")  # type: ignore[attr-defined]
                base.result = (
                    f"Mann-Whitney U ({col} by {grp}): U={stat:.1f}, p={p:.4f}"
                )

            case "linear_regression":
                if len(analysis.columns) < 2:
                    base.warning = (
                        "Linear regression requires at least 2 columns"
                    )
                    base.result = "Skipped"
                    return base
                outcome_col = analysis.columns[0]
                predictor_cols = analysis.columns[1:]
                for col in [outcome_col, *predictor_cols]:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        base.warning = f"Column '{col}' is not numeric"
                        base.result = "Skipped"
                        return base
                valid = df[[outcome_col, *predictor_cols]].dropna()
                if len(valid) < _MIN_SAMPLES["linear_regression"]:
                    base.warning = f"Too few observations ({len(valid)}) for regression"
                    base.result = "Skipped"
                    return base
                if len(valid) <= len(predictor_cols):
                    base.warning = "Need more observations than predictors"
                    base.result = "Skipped"
                    return base
                import statsmodels.api as sm
                x = sm.add_constant(valid[predictor_cols])
                model = sm.OLS(valid[outcome_col], x).fit()
                lines = [
                    f"Linear regression: {outcome_col} ~ {' + '.join(predictor_cols)}",
                    f"  R²={model.rsquared:.4f}, Adjusted R²={model.rsquared_adj:.4f}, "
                    f"F={model.fvalue:.3f}, p={model.f_pvalue:.4f}",
                ]
                for pred in predictor_cols:
                    coef = model.params[pred]
                    p_val = model.pvalues[pred]
                    lines.append(f"  {pred}: coef={coef:.4f}, p={p_val:.4f}")
                base.result = "\n".join(lines)

            case "logistic_regression":
                if len(analysis.columns) < 2:
                    base.warning = (
                        "Logistic regression requires at least 2 columns"
                    )
                    base.result = "Skipped"
                    return base
                outcome_col = analysis.columns[0]
                predictor_cols = analysis.columns[1:]
                # Encode binary outcome as 0/1
                outcome = df[outcome_col].dropna()
                unique_vals = outcome.unique()
                if len(unique_vals) != 2:
                    base.warning = (
                        f"Outcome '{outcome_col}' must have exactly "
                        f"2 values, found {len(unique_vals)}"
                    )
                    base.result = "Skipped"
                    return base
                label_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                valid = df[[outcome_col, *predictor_cols]].dropna()
                if len(valid) < _MIN_SAMPLES["logistic_regression"]:
                    base.warning = f"Too few observations ({len(valid)}) for logistic regression"
                    base.result = "Skipped"
                    return base
                y = valid[outcome_col].map(label_map).astype(float)
                for col in predictor_cols:
                    if not pd.api.types.is_numeric_dtype(valid[col]):
                        base.warning = f"Predictor '{col}' is not numeric"
                        base.result = "Skipped"
                        return base
                import statsmodels.api as sm
                x = sm.add_constant(valid[predictor_cols])
                model = sm.Logit(y, x).fit(disp=0)
                lines = [
                    f"Logistic regression: {outcome_col} ~ {' + '.join(predictor_cols)}",
                    f"  Pseudo R²={model.prsquared:.4f}, "
                    f"Log-Likelihood={model.llf:.1f}, AIC={model.aic:.1f}",
                    f"  Outcome mapping: {unique_vals[0]}=0, {unique_vals[1]}=1",
                ]
                for pred in predictor_cols:
                    coef = model.params[pred]
                    p_val = model.pvalues[pred]
                    odds = np.exp(coef)
                    lines.append(
                        f"  {pred}: coef={coef:.4f}, p={p_val:.4f}, odds_ratio={odds:.3f}"
                    )
                base.result = "\n".join(lines)

            case "two_way_anova":
                if len(analysis.columns) < 1:
                    base.warning = "Two-way ANOVA requires an outcome column"
                    base.result = "Skipped"
                    return base
                outcome_col = analysis.columns[0]
                if not pd.api.types.is_numeric_dtype(df[outcome_col]):
                    base.warning = f"Outcome '{outcome_col}' is not numeric"
                    base.result = "Skipped"
                    return base
                # Need exactly 2 grouping columns — use grouping_column + columns[1]
                if analysis.grouping_column is None or len(analysis.columns) < 2:
                    base.warning = (
                        "Two-way ANOVA requires 2 grouping columns"
                    )
                    base.result = "Skipped"
                    return base
                factor1 = analysis.grouping_column
                factor2 = analysis.columns[1]
                for f in [factor1, factor2]:
                    if f not in df.columns:
                        base.warning = f"Column '{f}' not found"
                        base.result = "Skipped"
                        return base
                valid = df[[outcome_col, factor1, factor2]].dropna()
                if len(valid) < _MIN_SAMPLES["two_way_anova"]:
                    base.warning = f"Too few observations ({len(valid)})"
                    base.result = "Skipped"
                    return base
                import statsmodels.api as sm
                from statsmodels.formula.api import ols as sm_ols
                formula = f"Q('{outcome_col}') ~ C(Q('{factor1}')) * C(Q('{factor2}'))"
                model = sm_ols(formula, data=valid).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                lines = [f"Two-way ANOVA: {outcome_col} ~ {factor1} × {factor2}"]
                for idx_name in anova_table.index:
                    if idx_name == "Residual":
                        continue
                    row = anova_table.loc[idx_name]
                    f_val = row.get("F", float("nan"))
                    p_val = row.get("PR(>F)", float("nan"))
                    label = idx_name.replace("C(Q('", "").replace("'))", "")
                    if pd.notna(f_val):
                        lines.append(f"  {label}: F={f_val:.3f}, p={p_val:.4f}")
                base.result = "\n".join(lines)

            case "pca":
                if len(analysis.columns) < 2:
                    base.warning = "PCA requires at least 2 numeric columns"
                    base.result = "Skipped"
                    return base
                for col in analysis.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        base.warning = f"Column '{col}' is not numeric"
                        base.result = "Skipped"
                        return base
                valid = df[analysis.columns].dropna()
                if len(valid) < _MIN_SAMPLES["pca"]:
                    base.warning = f"Too few observations ({len(valid)})"
                    base.result = "Skipped"
                    return base
                # Check for constant columns
                constant_cols = [c for c in analysis.columns if valid[c].std() == 0]
                if constant_cols:
                    base.warning = f"Constant column(s): {', '.join(constant_cols)}"
                    base.result = "Skipped"
                    return base
                from statsmodels.multivariate.pca import (
                    PCA as SM_PCA,
                )
                pca = SM_PCA(valid, standardize=True, demean=True)
                var_explained = pca.eigenvals / pca.eigenvals.sum()
                cum_var = np.cumsum(var_explained)
                lines = [f"PCA on {len(analysis.columns)} columns ({len(valid)} observations):"]
                for i, (v, c) in enumerate(zip(var_explained, cum_var)):
                    lines.append(
                        f"  PC{i + 1}: {v:.1%} variance (cumulative: {c:.1%})"
                    )
                    if c >= 0.95:
                        lines.append(f"  → {i + 1} components explain ≥95% of variance")
                        break
                base.result = "\n".join(lines)

    except Exception as exc:
        logger.opt(exception=True).debug("test {} failed", analysis.test)
        base.warning = f"Test failed: {exc}"
        base.result = "Skipped"

    return base


def _get_groups(
    df: pd.DataFrame, value_col: str, group_col: str,
) -> list[np.ndarray[Any, Any]] | str:
    """Split a numeric column into groups by a grouping column.

    Returns list of arrays on success, or an error message string on failure.
    """
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return f"Column '{value_col}' is not numeric"
    grouped = df.groupby(group_col)[value_col].apply(
        lambda x: x.dropna().to_numpy(),
    )
    groups = [g for g in grouped if len(g) > 0]
    return groups


# ---------------------------------------------------------------------------
# LLM test selection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research methods expert selecting statistical tests for a student's dataset.

You see column headers, data types, a sample of the data, and summary statistics.
Select tests from the provided palette ONLY. Do not invent tests.

Basic tests (appropriate for all students):
- shapiro: Shapiro-Wilk normality test. Use before parametric tests.
- t_test: Independent t-test. Use for 2 groups, numeric outcome, assumes normality.
- one_way_anova: One-way ANOVA. Use for 3+ groups, numeric outcome.
- tukey_hsd: Tukey HSD post-hoc. Use after significant ANOVA.
- chi_square: Chi-square test of independence. Use for 2 categorical variables.
- pearson_correlation: Pearson correlation. Use for 2 numeric variables, linear relationship.
- mann_whitney: Mann-Whitney U. Use for 2 groups, non-normal or ordinal data.

Intermediate tests (for undergraduate students and above):
- linear_regression: OLS regression. Use when predicting a numeric outcome from 1+ numeric \
predictors. First column in "columns" is the outcome, rest are predictors. Reports R², \
coefficients, and p-values.
- logistic_regression: Logistic regression. Use when outcome is binary (2 values). \
First column in "columns" is the binary outcome, rest are numeric predictors. Reports \
odds ratios.
- two_way_anova: Two-way ANOVA with interaction. Use when 2 categorical factors and \
1 numeric outcome. First column in "columns" is the outcome, second is the second factor. \
grouping_column is the first factor.

Advanced tests (for graduate students and professionals):
- pca: Principal Component Analysis. Use when 3+ numeric columns to understand \
dimensionality. Put all numeric columns in "columns". Reports variance explained per component.

Guidelines:
- Consider the student's level when selecting tests. Use basic tests by default. Only \
use intermediate/advanced tests if the student's profile indicates sufficient background.
- Consider variable types (numeric vs categorical), number of groups, and the research question.
- Include Shapiro-Wilk before parametric tests if sample size is reasonable (< 5000).
- If no tests are appropriate (e.g., data is just labels, single column), set skip_reason.
- Select only tests that make sense for the data structure. Don't force tests.
- grouping_column is required for t_test, one_way_anova, tukey_hsd, mann_whitney, and two_way_anova.
- For pearson_correlation and chi_square, put both columns in the columns list.
- For linear_regression and logistic_regression, first column is outcome, rest are predictors."""


def _build_pandas_summary(df: pd.DataFrame) -> list[str]:
    """Build a compact pandas summary for LLM context.

    Includes: numeric describe(), categorical value counts, and strong correlations.
    Mirrors the summary in text_extraction._extract_csv but kept independent
    to avoid coupling the analysis pipeline to the extraction pipeline.
    """
    parts: list[str] = []

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        desc = numeric_cols.describe().round(2)
        lines = []
        for col in desc.columns:
            s = desc[col]
            lines.append(
                f"  {col}: mean={s['mean']}, std={s['std']}, "
                f"min={s['min']}, max={s['max']}"
            )
        parts.append("Numeric summary:\n" + "\n".join(lines))

    # Categorical summary (unique counts + top values)
    cat_cols = df.select_dtypes(exclude="number")
    if not cat_cols.empty:
        lines = []
        for col in cat_cols.columns:
            n_unique = df[col].nunique()
            counts = df[col].value_counts().head(5)
            top = ", ".join(f"{v} ({c})" for v, c in counts.items())
            lines.append(f"  {col}: {n_unique} unique — {top}")
        parts.append("Categorical summary:\n" + "\n".join(lines))

    # Strong correlations
    if len(numeric_cols.columns) >= 2:
        corr = numeric_cols.corr()
        strong: list[str] = []
        seen: set[tuple[str, str]] = set()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i >= j:
                    continue
                r = corr.iloc[i, j]
                if abs(r) > 0.3 and (c1, c2) not in seen:
                    seen.add((c1, c2))
                    strong.append(f"  {c1} ↔ {c2}: r={r:.2f}")
        if strong:
            parts.append("Correlations (|r| > 0.3):\n" + "\n".join(strong))

    return parts


def _build_dataset_description(
    df: pd.DataFrame,
    description: str | None = None,
    research_question: str | None = None,
    student_profile: dict[str, Any] | None = None,
) -> str:
    """Build the user message describing the dataset for the LLM."""
    col_info = []
    for col in df.columns:
        dtype = df[col].dtype
        kind = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
        n_unique = df[col].nunique()
        n_missing = int(df[col].isna().sum())
        col_info.append(f"- {col} ({kind}, {n_unique} unique, {n_missing} missing)")

    sample = df.head(5).to_markdown(index=False)

    parts = [
        f"Dataset: {len(df)} rows × {len(df.columns)} columns\n",
        "Columns:\n" + "\n".join(col_info) + "\n",
        f"Sample (first 5 rows):\n{sample}",
    ]

    # Pandas summary — gives the LLM distributional context for test selection
    summary_parts = _build_pandas_summary(df)
    if summary_parts:
        parts.append("\n" + "\n".join(summary_parts))

    # Student profile — helps LLM pick level-appropriate tests
    if student_profile:
        profile_parts = _format_student_profile(student_profile)
        if profile_parts:
            parts.append(f"\nStudent profile:\n{profile_parts}")

    if description:
        parts.append(
            f"\n<user_content>Student's description: {description}</user_content>"
        )
    if research_question:
        parts.append(
            f"\n<user_content>Research question: {research_question}</user_content>"
        )

    return "\n".join(parts)


def _format_student_profile(profile: dict[str, Any]) -> str:
    """Format student profile fields for the LLM prompt."""
    lines = []
    if profile.get("education_level"):
        lines.append(f"- Education level: {profile['education_level']}")
    if profile.get("age"):
        lines.append(f"- Age: {profile['age']}")
    if profile.get("grade"):
        lines.append(f"- Grade: {profile['grade']}")
    if profile.get("college_year"):
        lines.append(f"- College year: {profile['college_year']}")
    if profile.get("domain_expertise"):
        expertise = profile["domain_expertise"]
        if isinstance(expertise, list) and expertise:
            lines.append(f"- Domain expertise: {', '.join(expertise)}")
    if profile.get("professional_experience"):
        exp = profile["professional_experience"]
        if isinstance(exp, list) and exp:
            lines.append(f"- Professional experience: {', '.join(exp)}")
    if profile.get("background_notes"):
        lines.append(f"- Background: {profile['background_notes']}")
    return "\n".join(lines)


async def select_tests(
    df: pd.DataFrame,
    description: str | None = None,
    research_question: str | None = None,
    prior_errors: str | None = None,
    student_profile: dict[str, Any] | None = None,
) -> AnalysisPlan:
    """Ask the LLM to select appropriate statistical tests for the dataset.

    If ``prior_errors`` is provided, the LLM is told which tests failed
    and why so it can make better selections on retry.

    If ``student_profile`` is provided, the LLM uses education level and
    background to decide which tier of tests is appropriate.
    """
    dataset_desc = _build_dataset_description(
        df, description, research_question, student_profile,
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=dataset_desc),
    ]

    if prior_errors:
        messages.append(HumanMessage(
            content=(
                "The previous analysis plan had errors. "
                "Please select different tests that avoid these issues:\n\n"
                + prior_errors
            ),
        ))

    return await structured_call(AnalysisPlan, messages, thinking="low")


# ---------------------------------------------------------------------------
# Public API — full pipeline
# ---------------------------------------------------------------------------


_MAX_RETRIES = 3


async def analyze_dataframe(
    df: pd.DataFrame,
    description: str | None = None,
    research_question: str | None = None,
    student_profile: dict[str, Any] | None = None,
) -> tuple[str, list[AnalysisResult]]:
    """Run the full analysis pipeline on a dataframe.

    If all or most tests fail, feeds the errors back to the LLM for
    up to 3 attempts. After exhausting retries, reports whatever succeeded.

    Returns (formatted_text, results_list).
    The formatted text is suitable for storing in the DB and embedding.
    """
    prior_errors: str | None = None

    for attempt in range(_MAX_RETRIES):
        plan = await select_tests(
            df, description, research_question,
            prior_errors=prior_errors, student_profile=student_profile,
        )

        if plan.skip_reason:
            text = f"## Statistical Analysis\n\nNo tests selected: {plan.skip_reason}"
            return text, []

        results: list[AnalysisResult] = []
        for analysis in plan.analyses:
            result = _run_test(analysis, df)
            results.append(result)

        completed = [r for r in results if r.warning is None]
        skipped = [r for r in results if r.warning is not None]

        # If at least one test succeeded, accept the results
        if completed:
            return format_results(results), results

        # All tests failed — build error feedback for next attempt
        if attempt < _MAX_RETRIES - 1:
            error_lines = [
                f"- {r.test}: {r.warning}" for r in skipped
            ]
            prior_errors = "\n".join(error_lines)
            logger.debug(
                "All {} tests failed on attempt {}/{}, retrying with error feedback",
                len(skipped), attempt + 1, _MAX_RETRIES,
            )

    # Exhausted retries — report what we have (all skipped)
    return format_results(results), results


def format_results(results: list[AnalysisResult]) -> str:
    """Format analysis results as human-readable text."""
    if not results:
        return ""

    lines = ["## Statistical Analysis"]

    completed = [r for r in results if r.warning is None]
    skipped = [r for r in results if r.warning is not None]

    for r in completed:
        lines.append(f"\n{r.result}")
        lines.append(f"  _{r.reasoning}_")

    if skipped:
        lines.append(f"\n{len(skipped)} test(s) skipped:")
        for r in skipped:
            lines.append(f"- {r.test}: {r.warning}")

    summary = f"\n{len(completed)} of {len(results)} analyses completed"
    if skipped:
        summary += f", {len(skipped)} skipped"
    lines.append(summary)

    return "\n".join(lines)
