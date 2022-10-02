#!/usr/bin/bash

# This script provides 3 ways to specify options to the vuze_merge.py script.
# 1. Arguments provided to the bash script: convert_list.sh list.dat --quality 720
# 2. Arguments provided for each item in the list: 102,0035,--quality 720,--format gpano
# 3. An existing config_102_0035.dat file with options readable by vuze_merge.dat
# note: The options 1, 2, 3 above are ordered by priority.
#       Options provided using method 1 will always take priority over 2 and 3.

list_file="$1"
args=("$@")
args=("${args[@]:1}")

while IFS= read -r line
do
    split=(${line//,/ })
    dir="${split[0]}"
    pic="${split[1]}"

    # check if the line is commented out, comments start with #
    [[ $dir =~ ^#.* ]] && continue

    addl=("${split[@]:2}")

    echo "${dir}HETVR/HET_${pic}"
    if test -f "config_${dir}_${pic}.dat"; then
        ../project/src/vuze_merge.py -r coeffs_v1_color.dat -c "config_${dir}_${pic}.dat" ${addl[@]} ${args[@]}
    else
        ../project/src/vuze_merge.py -f gpano,over-under -r coeffs_v1_color.dat -I "./${dir}HETVR/HET_${pic}" -O "HET${dir}_${pic}" ${addl[@]} ${args[@]}
    fi
done < "$list_file"
