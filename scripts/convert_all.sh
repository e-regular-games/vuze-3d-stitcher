#!/usr/bin/bash

overwrite=0
args=("$@")

for a in "${args[@]}"
do
    if [ "$a" == "--overwrite" ]; then
        overwrite=1
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
        project/src/vuze_merge.py -f over-under -r raw/coeffs_v1_color.dat -I "raw/${dir}HETVR/HET_${pic}" -O "stitched/HET${dir}_${pic}" ${args[@]}
    done
done
