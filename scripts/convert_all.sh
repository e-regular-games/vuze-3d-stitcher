#!/usr/bin/bash

overwrite=0
args=("$@")
args_pass=()

for a in "${args[@]}"
do
    if [ "$a" == "--overwrite" ]; then
        overwrite=1
    else
        args_pass+=("${a}")
    fi
done

for d in raw/*HETVR
do
    dir="${d:4:3}"
    for f in "$d"/HET_*.CSV
    do
        pic="${f:17:4}"
        if test -f "stitched/HET${dir}_${pic}.JPG" && [ "${overwrite}" != 1 ]; then
            continue
        fi

        echo "raw/${dir}HETVR/HET_${pic}"
        project/src/vuze_merge.py -f over-under -r raw/coeffs_v4.dat -I "raw/${dir}HETVR/HET_${pic}" -O "stitched/HET${dir}_${pic}" ${args_pass[@]}
    done
done
