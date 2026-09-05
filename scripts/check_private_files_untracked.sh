#!/usr/bin/env bash
# Fail if a file that must stay private is tracked in git.
#
# Both CI workflows call this instead of inlining the list. The list was
# duplicated in ci.yml and security.yml and drifted the first time a file was
# published: ci.yml was updated, security.yml was not, and the PR failed on a
# stale copy. One list, one place to edit.
set -euo pipefail

PRIVATE_FILES=(
  config.py
  data_pipeline.py
  value_betting_engine.py
  models/position_specific/weekly.py
)

tracked=$(git ls-files -- "${PRIVATE_FILES[@]}")
if [ -n "$tracked" ]; then
  echo "::error::Private files are tracked in git: ${tracked//$'\n'/ }"
  exit 1
fi
