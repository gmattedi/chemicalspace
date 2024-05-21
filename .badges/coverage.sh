#!/bin/bash
set -e

root=$(git rev-parse --show-toplevel)
total=$(coverage report --format=total --skip-empty --data-file="${root}"/.coverage)

function coverage_color() {
    echo $1 | awk '{
        if ($1 < 50) {
            print "red"
        } else if ($1 < 80) {
            print "yellow"
        } else {
            print "brightgreen"
        }
    }'
}
color=$(coverage_color "${total}")

curl -s https://img.shields.io/badge/coverage-"${total}"%25-"${color}" >"${root}"/.badges/coverage.svg
