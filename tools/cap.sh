#!/bin/sh
# Captures a species against the key backdrop and under the comparison light, for each
# seed in SEEDS (default 1 2 3), and scores the lot against the references.
#   SEEDS="1 2" [VIEWER=path/to/arbor-viewer.exe] tools/cap.sh <species> <refs-key> <tag> [viewer args...]
sp=$1; key=$2; tag=$3; shift 3
mkdir -p out/iter
light="--hdri sky --sun-elevation 45 --sun-azimuth 120 --neutral-sun --sun-intensity 0.7 --ground none --wind 0 --pitch 0.05 --bloom 0"
caps=""
for s in ${SEEDS:-1 2 3}; do
  "${VIEWER:-./target/release/arbor-viewer.exe}" --species "$sp" --seed $s --screenshot "out/iter/${tag}_s$s.png" \
    $light --backdrop 1,0,1 "$@" >/dev/null 2>&1
  "${VIEWER:-./target/release/arbor-viewer.exe}" --species "$sp" --seed $s --screenshot "out/iter/${tag}_s${s}_look.png" \
    $light --backdrop 0.8,0.8,0.82 "$@" >/dev/null 2>&1
  caps="$caps out/iter/${tag}_s$s.png"
done
python -P tools/refmatch.py "$key" $caps --debug "out/iter/${tag}_dbg.png"
