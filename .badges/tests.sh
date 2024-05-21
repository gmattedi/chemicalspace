#!/bin/bash
set -e

root=$(git rev-parse --show-toplevel)
curl -s https://img.shields.io/badge/tests-passing-brighgreen >"${root}"/.badges/tests.svg
