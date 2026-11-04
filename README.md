# ConvexIR

Code for the COMP6704 individual report *Convex Query Adaptation for Text Retrieval: Gradient Descent, Newton's Method, and Adam*.

## Overview

A dense retriever encodes queries and documents as vectors and ranks documents by their dot product with the query. Retraining the encoder changes every document vector, so the document index must be rebuilt. This project keeps the document vectors fixed and adapts only the query. A diagonal adapter multiplies each of the 384 query dimensions by a learned weight. When all weights equal 1, the retriever is the original one.

The weights are trained on triples of a query, a relevant document, and a non-relevant document, with a pairwise logistic loss. A penalty with strength λ pulls the weights toward 1. The resulting objective is strongly convex, so it has a unique minimizer, and the gradient norm bounds the distance to that minimizer. This bound serves as a common stopping rule for three solvers: gradient descent with backtracking line search, Newton's method with backtracking line search, and full-batch Adam.

Three experiments use the SciFact dataset and the `all-MiniLM-L6-v2` encoder:

- **Solver comparison** runs the three solvers for four values of λ and three random seeds, and records the number of updates and the solve time.
- **Sample-size comparison** repeats the solver comparison at λ = 1e-3 on subsets of 512 and 2,048 training triples.
- **Retrieval evaluation** ranks the full document collection with the solved weights and measures nDCG@10, which rewards relevant documents placed near the top of the first ten results. It selects systems on validation queries and then evaluates them once on test queries.

The repository contains code, configurations, tests, and version locks, but no data or results. The steps in [Running the project](#running-the-project) download the data and regenerate the results, figures, and tables of the report.

## Project structure

```text
comp6704-convex-ir/
├── configs/
│   ├── setup.toml                   # Dataset, encoder, query split, training triples
│   ├── adam-rates.toml              # Adam learning-rate selection
│   └── experiments.toml             # Solver and sample-size comparisons
├── src/query_adapt/
│   ├── cli.py                       # Command-line entry point for data preparation
│   ├── prepare.py                   # Download, encoding, baseline ranking, training triples
│   ├── data.py                      # Data loading, query split, triple sampling
│   ├── objective.py                 # Training objective, gradient, Hessian, stopping rule
│   ├── solvers.py                   # Gradient descent, Newton's method, Adam
│   ├── experiment.py                # Adam learning-rate selection
│   ├── campaign.py                  # Solver and sample-size comparisons
│   ├── retrieval.py                 # Full-collection ranking and retrieval metrics
│   ├── selection.py                 # Validation selection rule and bootstrap intervals
│   ├── evaluation.py                # Retrieval evaluation on validation and test queries
│   └── pilot.py                     # Logging and status helpers, and the development pilot run
├── tools/
│   ├── protocol.py                  # Experiment plan and timing procedure
│   ├── calibration_report.py        # Summary of the learning-rate selection
│   ├── posthoc_analysis.py          # Retrieval metrics on the training queries
│   └── report_figures.py            # Figures and tables of the report
├── scripts/
│   └── reproduce.sh                 # Runs every experiment, then draws the figures and tables
├── tests/                           # Unit tests on synthetic data
├── environment.yml                  # Conda environment, any platform
├── conda-explicit.txt               # Exact conda packages, macOS arm64
├── requirements-lock.txt            # Exact Python packages
├── requirements-plot.txt            # Plotting packages
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/DannyWang1922/comp6704-convex-ir.git
cd comp6704-convex-ir
conda create -n comp6704 --file conda-explicit.txt
conda activate comp6704
python -m pip install -r requirements-lock.txt -r requirements-plot.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

The locked versions are those used for the report on macOS arm64, with Python 3.11.18, NumPy 2.4.6, and SciPy 1.17.1. On other platforms, replace the `conda create` line with `conda env create -f environment.yml`.

## Running the project

With the environment active, run from the repository root of a fresh clone:

```bash
bash scripts/reproduce.sh
```

The script runs six steps in order and stops at the first error:

1. **Prepare the data.** Downloads SciFact and the encoder, encodes all documents and queries, splits the queries, and builds the training triples. Output: `artifacts/`.
2. **Select the Adam learning rates.** Builds the training triples for seeds 42, 43, and 44, and selects the Adam learning rate for each λ on the seed 42 triples. Output: `artifacts/runs/adam-rates/`.
3. **Solver comparison.** Solves the 12 problems given by four values of λ and three seeds with each of the three solvers, and times every solve. Output: `artifacts/runs/full-experiment-001/e1-summary.json` and `e1-timings.csv`.
4. **Sample-size comparison.** Repeats the comparison at λ = 1e-3 and seed 42 on subsets of 512 and 2,048 triples, reusing the full set from step 3. Output: `artifacts/runs/full-experiment-001/e3-summary.json` and `e3-timings.csv`.
5. **Retrieval evaluation.** Ranks the full collection with the original retriever and with the solved weights for each λ. It chooses systems on the validation queries, records the choice, and then evaluates the chosen systems once on the test queries. Output: `artifacts/runs/full-experiment-001/e2/`.
6. **Figures and tables.** Draws the figures and tables of the report from the saved results. Output: `artifacts/report/`.

The commands of each step are listed in `scripts/reproduce.sh` and can also be run one at a time.

## Experimental settings

| Setting | Value |
| --- | --- |
| Dataset | SciFact from BEIR, downloaded from `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`; 5,183 documents |
| Query split | 809 queries with training labels, split by seed 42 into 647 training and 162 validation queries; 300 test queries |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; title and text, truncated at 256 word pieces; normalized 384-dimensional embeddings |
| Training triples | For each of the 742 relevant training pairs, 4 hard negatives from the original ranking and 4 random negatives; 5,936 triples; score temperature 0.1 |
| Random seeds | 42, 43, 44, which change the random negatives only |
| Regularization λ | 1e-2, 1e-3, 1e-4, 1e-5 |
| Stopping rule | Both `‖∇F‖² / (2λ) ≤ 1e-9` and `‖∇F‖ / λ ≤ 1e-3`; at most 20,000 updates or 60 s per solve |
| Initial weights | All equal to 1 |
| Gradient descent | Backtracking line search with sufficient-decrease constant 0.5 and shrink factor 0.5; each trial step is twice the previous accepted step |
| Newton's method | Exact Hessian; backtracking line search from step 1 with sufficient-decrease constant 1e-4 and shrink factor 0.5 |
| Adam | β1 = 0.9, β2 = 0.999, ε = 1e-8; learning rates 0.01, 0.1, 0.3, 0.3 for λ = 1e-2, 1e-3, 1e-4, 1e-5 |
| Adam learning-rate selection | On the seed 42 triples, the rate from {0.003, 0.01, 0.03} that meets the stopping rule in the fewest updates, extended to 0.1 and 0.3 when the largest rate is best |
| Sample sizes | Nested subsets of 512 and 2,048 seed 42 triples, drawn with seed 2026, plus the full 5,936 |
| Timing | One CPU thread; one warmup and five timed blocks per solver, with solver order rotated between blocks; a block averages 10 solves when the warmup takes under 20 ms |
| Retrieval weights | For each λ and seed, a Newton solution with the tighter stopping bounds 1e-12 and 1e-5 |
| Selection rule | A λ is eligible when its mean validation nDCG@10 is within one paired-bootstrap standard error of the best system. The original retriever is preferred, then the largest eligible λ. The best λ is also tested as a secondary comparison. |
| Metrics and intervals | nDCG@10 and Recall@100; paired 95% percentile intervals from 2,000 query-level bootstrap samples |

The configuration files in `configs/` set the data, triples, λ grid, stopping rule, and solver parameters. The subset seed, timing procedure, and selection rule are fixed in `tools/protocol.py` and `src/query_adapt/selection.py`.
