#!/bin/bash

# Script that runs all bash scripts in the .badges directory to update the badges

set -e

script_basename=$(basename "$0")

root=$(git rev-parse --show-toplevel)

for badge in "${root}"/.badges/*.sh; do
    # skip itself
    if [[ "${badge}" == "${root}"/.badges/"${script_basename}" ]]; then
        continue
    fi

    bash "${badge}"
done
