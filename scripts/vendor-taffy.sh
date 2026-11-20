#!/usr/bin/env bash
# Re-makes vendor/taffy: the taffy crate exactly as published on crates.io at <version>, plus
# vendor/taffy.patch, which measures a flex item's automatic minimum size (`min-width: auto`) only
# when it can change the layout. Taffy measures every item's min-content size up front; an item
# whose own main size is definite can only be held at that minimum when its line shrinks, so most
# of those measurements (a whole card's subtree, for each card a row of them mounts) went unused.
# Debug builds still measure every one up front and check the result against it.
# Run it when taffy is bumped, then build: a patch that no longer applies fails here, and the
# changes go back in by hand.
# Usage: scripts/vendor-taffy.sh <version>     e.g. scripts/vendor-taffy.sh 0.15.1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
version="${1:?usage: scripts/vendor-taffy.sh <version>}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "https://static.crates.io/crates/taffy/taffy-$version.crate" -o "$tmp/crate.tar.gz"
tar -xzf "$tmp/crate.tar.gz" -C "$tmp"
rm -rf vendor/taffy
mv "$tmp/taffy-$version" vendor/taffy
# The published crate leaves out the MIT license text; take it from the release's tag.
curl -fsSL "https://raw.githubusercontent.com/DioxusLabs/taffy/v$version/LICENSE" -o vendor/taffy/LICENSE
if [ -s vendor/taffy.patch ]; then git apply --directory=vendor/taffy vendor/taffy.patch; fi
echo "vendor/taffy: taffy $version + vendor/taffy.patch"
