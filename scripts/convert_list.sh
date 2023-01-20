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

overwrite=0
args_pass=()

# returns an array of strings, each string is a cell, cells may include commas
# if the contents were originally quoted.
function parse_csv {
    line=$2
    splitq=(${line//\"/ })
    i=0

    local -n res=$1
    res=()
    for s in "${splitq[@]}"
    do
        if [[ $(expr $i % 2) == "0" ]]; then
            res+=( ${s//,/ } )
        else
            res+=( ${s} )
        fi
        i=$((i+1))
    done
}

for a in "${args[@]}"
do
    if [ "$a" == "--overwrite" ]; then
        overwrite=1
    else
        args_pass+=("${a}")
    fi
done

list_dir_full=(${list_file//./ })
list_dir_arr=(${list_dir_full[0]//\// })
list_dir=${list_dir_arr[1]}

mkdir -p "stitched/${list_dir}"

while IFS= read -r line
do
    split=()
    parse_csv split "$line"

    if [ "${#split[@]}" -lt "2" ]; then
        continue
    fi

    dir="${split[0]}"
    pic="${split[1]}"

    # check if the line is commented out, comments start with #
    [[ $dir =~ ^#.* ]] && continue

    addl=("${split[@]:2}")

    if test -f "stitched/${list_dir}/HET${dir}_${pic}.JPG" && [ "${overwrite}" != 1 ]; then
        continue
    fi

    echo "${dir}HETVR/HET_${pic}"
    if test -f "raw/config_${dir}_${pic}.dat"; then
        project/src/vuze_merge.py -f over-under,gpano -a project/coeffs_v5.dat -c "raw/config_${dir}_${pic}.dat" -O "stitched/${list_dir}/HET${dir}_${pic}" ${addl[@]} ${args_pass[@]}
    else
        project/src/vuze_merge.py -f over-under,gpano -a project/coeffs_v5.dat -I "raw/${dir}HETVR/HET_${pic}" -O "stitched/${list_dir}/HET${dir}_${pic}" ${addl[@]} ${args_pass[@]}
    fi
done < "$list_file"
