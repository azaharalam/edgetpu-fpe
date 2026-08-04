#!/usr/bin/env bash
# Models are not vendored -- they come from Google's Coral test_data repo.
set -e
cd "$(dirname "$0")/.."
[ -d test_data ] || git clone --depth 1 https://github.com/google-coral/test_data.git
echo "models in $(pwd)/test_data"
