#!/bin/sh
# Assemble the playground's Docker build context (also the Hugging Face Space contents).
#   deploy/huggingface/assemble.sh <out-dir>
set -eu
out=${1:?usage: assemble.sh <out-dir>}
root=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$out"
mkdir -p "$out/deploy"
cp "$root/pyproject.toml" "$out/"
cp "$root/deploy/huggingface/Dockerfile" "$root/deploy/huggingface/README.md" "$out/"
cp "$root/deploy/playground.toml" "$out/deploy/"
# Tracked files only: no caches, local state or anything git ignores.
git -C "$root" archive --format=tar HEAD src | tar -x -C "$out"
echo "assembled $(find "$out" -type f | wc -l) files in $out"
