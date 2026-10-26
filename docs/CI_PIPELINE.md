# CI Pipeline

The GitHub Actions workflow is defined in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It runs on pushes to
`main` or `develop` and on pull requests targeting either branch. The workflow
uses Python 3.11 and caches pip downloads based on `requirements.txt`.

## Jobs

The workflow runs three jobs in dependency order:

```text
build-and-lint -> test-suite -> model-quality-gate
```

### `build-and-lint`

Installs project requirements, compiles `src/` and `tests/`, and runs Flake8
over `src/` with a 100-character line limit. `E203` is ignored for compatibility
with common formatting conventions. The intentional imports after
`matplotlib.use("Agg")` in the MLflow training module are locally marked with
`# noqa: E402`, since setting the non-interactive backend before importing
`pyplot` is required for headless CI runs.

### `test-suite`

Runs after `build-and-lint` succeeds. It creates the raw Iris CSV, preprocesses
it, engineers features, and runs the data validation CLI. It then runs the full
pytest suite with terminal coverage, Cobertura-style `coverage.xml`, and JUnit
`test-results.xml`. The two report files are uploaded as the
`test-and-coverage-reports` artifact, including when tests fail if the reports
were produced.

### `model-quality-gate`

Runs after `test-suite` succeeds and separately runs
`tests/test_model_quality.py`. That test uses scikit-learn's bundled Iris
dataset, checks valid predicted labels, and requires at least 0.90 held-out
accuracy. The separate job makes the model-quality result visible as a distinct
check on successful full-suite runs. If the quality assertion fails in
`test-suite`, this job is skipped because of its `needs: test-suite` dependency;
the failing test-suite check still blocks a merge when required.

## Requiring checks before merging

The workflow does not configure GitHub repository rules. To require CI before
merging, a repository administrator must configure a branch ruleset or branch
protection rule in GitHub Settings targeting `main`:

1. Require pull requests before merging.
2. Require status checks to pass.
3. Select the checks corresponding to `build-and-lint`, `test-suite`, and
   `model-quality-gate` after GitHub has observed a completed workflow run.
4. Optionally require the branch to be up to date before merging.

The actual check labels are shown in the repository's settings after the
workflow has run. Confirm the ruleset is active and targets `main`; these
settings must be verified in GitHub and are not changed by this repository
workflow.

## Verifying CI and reports

Push a change or open a pull request targeting `main` or `develop`, then inspect
the **MLOps CI Pipeline** run in the GitHub Actions tab. A successful run should
show all three jobs passing. Download the `test-and-coverage-reports` artifact
from the run summary to inspect `test-results.xml` and `coverage.xml`.

To verify the model-quality assertion locally, temporarily raise
`MIN_ACCEPTABLE_ACCURACY` in `tests/test_model_quality.py` from `0.90` to
`0.999` and run:

```bash
python -m pytest tests/test_model_quality.py -v
```

The quality test should fail because the current measured held-out accuracy is
approximately `0.9667`. Restore the threshold to `0.90` before committing or
merging. A failing assertion in the full test-suite job causes the downstream
model-quality job to be skipped; do not merge a branch with failed required
checks.

## Scope of verification

The workflow file and local lint, compile, pipeline, and pytest commands can be
verified from a checkout. Pull-request status checks, downloaded Actions
artifacts, and branch-protection enforcement require a workflow run and
repository settings on GitHub; do not treat them as confirmed until inspected
there.
