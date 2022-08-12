#!/bin/bash

# Requires:
# pfstools
# imagemagick
# exiftool
# luminance-hdr-cli
# enfuse

config_file=""
clean=1
verbose=0
montage=0

usage() {
    echo "Usage: $0 -c <config_file> [-d] [-v] [-m]" 1>&2;
    echo "d - leave temporary files behind (dirty)" 1>&2;
    echo "v - verbose" 1>&2;
    echo "m - create a montage of each lens (left image output)" 1>&2;
    echo "" 1>&2;
    echo "-- Config Format --" 1>&2;
    echo "out,<output_prefix>" 1>&2;
    echo "lens,<idx>,<center_x>,<center_y>,<radius>" 1>&2;
    echo "image,<exposure>,<image_prefix>" 1>&2;
    echo "hdr,method,<enfuse|luminance>" 1>&2;
    echo "hdr,clamp,<ratio>" 1>&2;
    echo "hdrl,<lens>,clamp,<ratio>" 1>&2;
    echo "hdr,negate,<0_or_1>" 1>&2;
    echo "hdr,autoag,<antighosting_threshold>" 1>&2;
    echo "ldr,<luminance_hdr_config>" 1>&2;
    echo "ldrl,<lens>,<luminance_hdr_config_per_lens>" 1>&2;
    echo "" 1>&2;
    echo "-- Config Example -- (abbreviated)" 1>&2;
    echo "out,HET_1000" 1>&2;
    echo "lens,1,544,778,734" 1>&2;
    echo "image,-4,HET_0011" 1>&2;
    echo "hdr,method,luminance" 1>&2;
    echo "hdr,clamp,0.985" 1>&2;
    echo "hdr,autoag,0.1" 1>&2;
    echo "ldr,TMOSETTINGSVERSION=0.6" 1>&2;
    echo "ldrl,1,TMOSETTINGSVERSION=0.6" 1>&2;
    echo "" 1>&2;
    echo "Notes: All config lines are required, 8 lenses are required, and at least 3 images" 1>&2;
    echo "enfuse: valid hdr/ldr options: negate" 1>&2;
    echo "luminance: valid hdr/ldr options: clamp, negate, autoag, ldr" 1>&2;
    exit 1;
}

exec_cmd() {
    res=0
    if (( verbose == 1 )); then
        echo "$@"
        eval $@
        res=$?
    else
        out=`eval $@ 2>&1`
        res=$?
    fi

    if [ ${res} -ne 0 ]; then
        echo "failed: $@"
        exit 1
    fi
}


while getopts "mvdc:" o; do
    case "${o}" in
        c)
            config_file=${OPTARG}
            ;;
        d)
            clean=0
            ;;
        m)
            montage=1
            ;;
        v)
            verbose=1
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "${config_file}" ]; then
    usage
fi

# the following arrays are indexed by lens number, 1..8
center_x=()
center_y=()
radius=()
base_dims=()

# indexed 0..N where N is the number of exposure brackets
exposure=()
image_base=()

ldr_config=""
ldrl_config=()
output=""
hdr_clamp=""
hdrl_clamp=()
hdr_args=()
hdr_negate=0
hdr_method="enfuse"
while IFS= read -r line
do
    cmd=(${line//,/ })
    cmd_len=${#cmd[@]}
    if [ "${cmd[0]}" = "lens" ] && [ ${cmd_len} -eq 5 ]; then
        l=${cmd[1]}
        center_x[$l]=${cmd[2]}
        center_y[$l]=${cmd[3]}
        radius[$l]=${cmd[4]}
    fi

    if [ "${cmd[0]}" = "image" ]; then
        exposure+=(${cmd[1]})
        image_base+=(${cmd[2]})
    fi

    if [ "${cmd[0]}" = "ldr" ]; then
        if [ -n "$ldr_config" ]; then
            ldr_config+="\n"
        fi
        ldr_config+="${cmd[1]}"
    fi

    if [ "${cmd[0]}" = "ldrl" ]; then
        lens=${cmd[1]}
        if [ -n "${ldrl_config[$lens]}" ]; then
            ldrl_config[$lens]+="\n"
        fi
        ldrl_config[$lens]+="${cmd[2]}"
    fi

    if [ "${cmd[0]}" = "hdrl" ] && [ "${cmd[2]}" = "clamp" ]; then
        lens=${cmd[1]}
        hdrl_clamp[$lens]=${cmd[3]}
    fi

    if [ "${cmd[0]}" = "out" ]; then
        output=${cmd[1]}
    fi

    if [ "${cmd[0]}" = "hdr" ] && [ "${cmd[1]}" = "clamp" ]; then
        hdr_clamp=${cmd[2]}
    elif [ "${cmd[0]}" = "hdr" ] && [ "${cmd[1]}" = "method" ]; then
        hdr_method=${cmd[2]}
    elif [ "${cmd[0]}" = "hdr" ] && [ "${cmd[1]}" = "negate" ]; then
        hdr_negate=${cmd[2]}
    elif [ "${cmd[0]}" = "hdr" ]; then
        hdr_args+=("--${cmd[1]}" "${cmd[2]}")
    fi
done < "$config_file"

if (( ${#exposure[@]} < 3 )); then
    echo "Must have at least 3 exposure brackets."
    exit 1
fi

if [ -z "$output" ]; then
    echo "An output file must be specified in the config."
    exit 1
fi

images=${#image_base[@]}
ref_idx=-1
for (( i=0; i<$images; i++ )); do
    base="${image_base[$i]}"
    exp="${exposure[$i]}"

    if (( ${exp} == 0 )); then
        ref_idx=$i
    fi

    base_dims=()
    # Rotate each image and crop based on the center_x, center_y, radius
    for lens in {1..8}; do
        file="${base}_${lens}.JPG"
        file_rot="${base}_${lens}_rot.JPG"
        file_crop="${base}_${lens}_crop.JPG"

        echo "pre-processing: ${file}"
        base_dims[$lens]=`identify $file | awk '{ print $3 }'`

        r=${radius[$lens]}
        diameter=$((r * 2))
        x=${center_x[$lens]}
        y=${center_y[$lens]}
        offset_y=$((y - r))

        exec_cmd convert ${file} -rotate 270 -crop "${diameter}x${diameter}+0+${offset_y}" +repage -quality 100% "${file_rot}"

        dims=`identify $file_rot | awk '{ print $3 }'`
        y=$((y - offset_y))
        width_img=`echo $dims | awk -F 'x' '{ print $1 }'`
        grow_left=$((r - x + width_img))

        exec_cmd convert -size "${dims}" xc:none -fill "${file_rot}" -draw "\"circle ${x},${y} ${x},1\"" -background "\"#808080\"" -gravity east -extent "${grow_left}x${diameter}" -gravity west -extent "${diameter}x${diameter}" -quality 100% "${file_crop}"

        if (( hdr_negate == 1 )); then
            exec_cmd mogrify -negate -quality 100% "${file_crop}"
        fi
    done
done

if (( $ref_idx == -1 )); then
    echo "One image must have exposure 0. That image will be the source of exif metadata."
    exit 1
fi

function merge_luminance() {
    lens=$1
    hdr_output=$2

    cfg="ldr_config${lens}.dat"

    if [ -z "${ldrl_config[$lens]}" ]; then
        echo -e "${ldr_config}" > "$cfg"
    else
        echo -e "${ldrl_config[$lens]}" > "$cfg"
    fi

    evs=""
    for ev in ${exposure[@]}; do
        if (( hdr_negate == 1 )); then ev=$(( -1 * ev )); fi
        if [ ! -z "$evs" ]; then evs+=","; fi
        evs+="${ev}"
    done

    hdr_make=(luminance-hdr-cli -e "\"${evs}\"" -s "hdr_${lens}.exr")
    hdr_make+=(${hdr_args[@]})
    for base in ${image_base[@]}; do
        hdr_make+=(${base}_${lens}_crop.JPG)
    done
    exec_cmd "${hdr_make[@]}"

    clamp=${hdrl_clamp[$lens]}
    if [ -z "${clamp}" ]; then
        clamp=${hdr_clamp}
    fi
    if [ ! -z "${clamp}" ]; then
        exec_cmd pfsin "hdr_${lens}.exr" "|" pfsclamp --max ${clamp} -p --rgb "|" pfsout "hdr_${lens}_clamped.exr"
    else
        exec_cmd cp "hdr_${lens}.exr" "hdr_${lens}_clamped.exr"
    fi

    exec_cmd luminance-hdr-cli -l "hdr_${lens}_clamped.exr" --tmofile "$cfg" -o "${hdr_output}" -q 100 --autolevels
}

function merge_enfuse() {
    lens=$1
    hdr_output=$2

    enfuse=(enfuse --output "hdr_${lens}.tif")
    for base in ${image_base[@]}; do
        exec_cmd convert "${base}_${lens}_crop.JPG" "${base}_${lens}_crop.tif"
        enfuse+=("${base}_${lens}_crop.tif")
    done

    exec_cmd ${enfuse[@]}

    exec_cmd convert "hdr_${lens}.tif" -quality 100% "${hdr_output}"
}

# Generate the exposure merged files.
for lens in {1..8}; do
    echo "merging exposure brackets: lens${lens}"

    if [ "${hdr_method}" = "enfuse" ]; then
        merge_enfuse ${lens} "hdr_${lens}.JPG"
    elif [ "${hdr_method}" = "luminance" ]; then
        merge_luminance ${lens} "hdr_${lens}.JPG"
    else
        echo "invalid merge method: ${hdr_method}" 1>&2
        exit 1
    fi

done

# write the ldr for each lens back to the correct place in the image
# and add the appropriate exif metdata
for lens in {1..8}; do
    file_hdr="hdr_${lens}.JPG"
    file_out="${output}_${lens}.JPG"
    file_ref="${image_base[$ref_idx]}_${lens}.JPG"

    r=${radius[$lens]}
    x=${center_x[$lens]}
    y=${center_y[$lens]}
    left=$((r - x))

    rot_width=`echo ${base_dims[$lens]} | awk -F 'x' '{ print $2 }'`
    rot_height=`echo ${base_dims[$lens]} | awk -F 'x' '{ print $1 }'`

    grow_top=$((y + r))
    exec_cmd convert "${file_hdr}" -crop "\"${rot_width}x${rot_height}+${left}+0\"" +repage -background "\"#808080\"" -gravity south -extent "${rot_width}x${grow_top}" -gravity north -extent "${rot_width}x${rot_height}" -rotate 90 -quality 100% "${file_out}"

    if (( hdr_negate == 1 )); then
        exec_cmd mogrify -negate -quality 100% "${file_out}"
    fi

    exec_cmd exiftool -overwrite_original -tagsFromFile "${file_ref}" "\"-all:all>all:all\"" "${file_out}"

    echo "output: ${file_out}"
done

if (( montage == 1 )); then
    for lens in {1..8}; do
        echo "montage: montage_${lens}.JPG"

        mm=(montage "${output}_${lens}.JPG")
        for base in ${image_base[@]}; do
            mm+=("${base}_${lens}.JPG")
        done
        mm+=(-geometry +0+0 -tile 1x -quality 100% "montage_v_${lens}.JPG")
        exec_cmd ${mm[@]}

        exec_cmd convert "montage_v_${lens}.JPG" -rotate 270 -quality 100% "montage_${lens}.JPG"
        rm "montage_v_${lens}.JPG"
    done
fi

if (( clean == 1 )); then
    for base in ${image_base[@]}; do
        r=`rm ${base}_*_rot.JPG 2>&1`
        r=`rm ${base}_*_crop.JPG 2>&1`
        r=`rm ${base}_*_crop.tif 2>&1`
    done
    for lens in {1..8}; do
        r=`rm "hdr_${lens}.exr" 2>&1`
        r=`rm "hdr_${lens}_clamped.exr" 2>&1`
        r=`rm "hdr_${lens}.JPG" 2>&1`
        r=`rm "hdr_${lens}.tif" 2>&1`
    done
    r=`rm ldr_config*.dat 2>&1`
fi
