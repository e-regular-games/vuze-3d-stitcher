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
        if test -f "preview/HET${dir}_${pic}.JPG" && [ "${overwrite}" != 1 ]; then
            continue
        fi

        echo "raw/${dir}HETVR/HET_${pic}"
        project/src/vuze_merge.py -a project/coeffs_v6.json -I "raw/${dir}HETVR/HET_${pic}" -O "preview/HET${dir}_${pic}" --format over-under --fast 3 --quality 720 ${args_pass[@]}
    done
done
