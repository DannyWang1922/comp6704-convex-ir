#!/usr/bin/env bash
# Run every experiment of the report, then draw its figures and tables.
# Usage, from a fresh clone with the comp6704 environment active:
#   bash scripts/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -e artifacts ]; then
  echo "artifacts/ already exists. Run this script in a fresh clone." >&2
  exit 1
fi

RATES=artifacts/runs/adam-rates
RUN=artifacts/runs/full-experiment-001
step() { printf '\n==> Step %s\n' "$1"; }

step "1/6: Prepare the data"
query-adapt download --config configs/setup.toml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 query-adapt encode --config configs/setup.toml
query-adapt baseline --config configs/setup.toml
query-adapt pairs --config configs/setup.toml
query-adapt verify --config configs/setup.toml

step "2/6: Select the Adam learning rates"
for stage in diagnostics freeze calibrate; do
  python -u -m query_adapt.experiment --config configs/adam-rates.toml --output "$RATES" --stage "$stage"
done
python -m query_adapt.campaign preflight --config configs/experiments.toml

step "3/6: Solver comparison"
python -u -m query_adapt.campaign run --stage e1 --config configs/experiments.toml --output "$RUN"

step "4/6: Sample-size comparison"
python -u -m query_adapt.campaign run --stage e3 --config configs/experiments.toml --output "$RUN"

step "5/6: Retrieval evaluation"
python -m query_adapt.evaluation manifest --campaign "$RUN" --manifest "$RUN/e2-manifest.json" \
  --embeddings artifacts/embeddings.npz --splits artifacts/data/splits.json \
  --train-qrels artifacts/data/scifact/qrels/train.tsv --test-qrels artifacts/data/scifact/qrels/test.tsv
for stage in validate freeze test summary; do
  python -m query_adapt.evaluation "$stage" --manifest "$RUN/e2-manifest.json" --output "$RUN/e2"
done

step "6/6: Figures and tables"
python tools/posthoc_analysis.py
python tools/report_figures.py

printf '\nDone. Experiment results are in %s; figures and tables are in artifacts/report/.\n' "$RUN"
