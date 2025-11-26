#!/usr/bin/bash

root="$1"
shift

for s in "$@"; do
	for c in CH HH BH LH; do
		for f in $(ls ${root}/${s}/${c}*/*Z* 2>/dev/null | head -1); do
			mseedinfo "$f"
		done
		for f in $(ls ${root}/${s}/${c}*/*Z* 2>/dev/null | tail -1); do
			mseedinfo "$f"
		done
	done
done
