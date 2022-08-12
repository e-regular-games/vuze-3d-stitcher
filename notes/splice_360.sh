#!/bin/bash

# Requires:
# imagemagick
# ffmpeg
# python
# exiftool

# lens 1: 544,778 734
# lens 2: 560,778 734
# lens 3: 500,800 734
# lens 4: 570,770 734
# lens 5: 490,830 734
# lens 6: 580,800 734
# lens 7: 544,778 734
# lens 8: 580,800 734


config_file=""
clean=1
verbose=0
montage=0

usage() {
    echo "Convert 8 individual lens images to a left and right eye 360 image." 1>&2;
    echo "  Translate fisheye to cubemap to assemble each eye and then converts to equirect." 1>&2;
    echo "Usage: $0 -c <config_file> [-d] [-v] [-m]" 1>&2;
    echo "d - leave temporary files behind (dirty)" 1>&2;
    echo "v - verbose" 1>&2;
    echo "m - create a montage of each lens (left image output)" 1>&2;
    echo "" 1>&2;
    echo "-- Config Format --" 1>&2;
    echo "in,<input_prefix>" 1>&2;
    echo "out,<output_prefix>" 1>&2;
    echo "radius,<radius>" 1&>2;
    echo "resolution,<resolution>" 1&>2;
    echo "lens,<idx>,<center_x>,<center_y>" 1>&2;
    echo "" 1>&2;
    echo "-- Config Example -- (abbreviated)" 1>&2;
    echo "in,HET_0011" 1>&2;
    echo "out,HET_1000" 1>&2;
    echo "lens,1,544,778,734" 1>&2;
    echo "radius,734" 1>&2;
    echo "resolution,1600" 1>&2;
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
radius=0
cube_faces=("" "rludfb" "rludfb" "fbudlr" "fbudlr" "lrudbf" "lrudbf" "bfudrl" "bfudrl")
cube_rot=("" "000000" "000000" "003100" "003100" "002200" "002200" "001300" "001300")

input=""
output=""
resolution=0

while IFS= read -r line
do
    cmd=(${line//,/ })
    cmd_len=${#cmd[@]}

    if [ "${cmd[0]}" = "lens" ] && [ ${cmd_len} -eq 4 ]; then
        l=${cmd[1]}
        center_x[$l]=${cmd[2]}
        center_y[$l]=${cmd[3]}
    fi

    if [ "${cmd[0]}" = "radius" ]; then
        radius=${cmd[1]}
    fi

    if [ "${cmd[0]}" = "resolution" ]; then
        resolution=${cmd[1]}
    fi

    if [ "${cmd[0]}" = "out" ]; then
        output=${cmd[1]}
    fi

    if [ "${cmd[0]}" = "in" ]; then
        input=${cmd[1]}
    fi

done < "$config_file"

if [ -z "$output" ]; then
    echo "An output file must be specified in the config."
    exit 1
fi

if [ -z "$input" ]; then
    echo "An input file must be specified in the config."
    exit 1
fi

if (( radius == 0 )); then
    echo "The radius in pixels of all lens must be defined."
    exit 1
fi

if (( resolution == 0 )); then
    echo "The resolution in pixels of the output must be defined."
    exit 1
fi

function generate_eq180() {
    lens=$1
    file="${input}_${lens}.JPG"
    file_ext="${file##*.}"
    file_name="${file%.*}"
    file_rot="spl_${file_name}_rot.${file_ext}"
    file_crop="spl_${file_name}_crop.${file_ext}"
    file_eq360="spl_${file_name}_eq360.${file_ext}"
    file_eq180="spl_${file_name}_eq180.${file_ext}"

    diameter=$((radius * 2))
    cx=${center_x[$lens]}
    cy=${center_y[$lens]}
    offset_y=$((cy - radius))

    exec_cmd convert ${file} -rotate 270 -crop "${diameter}x${diameter}+0+${offset_y}" +repage -quality 100% "${file_rot}"

    dims=`identify $file_rot | awk '{ print $3 }'`
    width_img=`echo $dims | awk -F 'x' '{ print $1 }'`
    height_img=`echo $dims | awk -F 'x' '{ print $2 }'`
    cy=$((cy - offset_y))
    grow_left=$((radius - cx + width_img))

    exec_cmd convert -size "${dims}" xc:none -fill "${file_rot}" -draw "\"circle ${cx},${cy} ${cx},1\"" -background "\"#808080\"" -gravity east -extent "${grow_left}x${diameter}" -gravity west -extent "${diameter}x${diameter}" -quality 100% "${file_crop}"

    #cube_h=$((resolution * 6))
    #exec_cmd ffmpeg -i "${file_crop}" -y -vf "v360=fisheye:c1x6:ih_fov=175:iv_fov=175:out_forder=${cube_faces[$lens]}:out_frot=${cube_rot[$lens]}:w=${resolution}:h=${cube_h}" -qmin 1 -qscale:v 1 "${file_cube}"

    eq_w=$((resolution * 2))
    exec_cmd ffmpeg -i "${file_crop}" -y -vf "v360=fisheye:equirect:ih_fov=175:iv_fov=175:h=${resolution}:w=${eq_w}" -qmin 1 -qscale:v 1 "${file_eq360}"

    eq180_offset=$((resolution / 2))
    exec_cmd convert ${file_eq360} -crop "${resolution}x${resolution}+${eq180_offset}+0" +repage -quality 100% "${file_eq180}"
}

function generate_cube_map() {
    lens=$1
    file="${input}_${lens}.JPG"
    file_ext="${file##*.}"
    file_name="${file%.*}"
    file_cube="spl_${file_name}_cube.${file_ext}"
    file_eq180="spl_${file_name}_eq180.${file_ext}"
    file_eq360="spl_${file_name}_eq360.${file_ext}"

    eq_w=$((resolution * 2))
    exec_cmd convert "${file_eq180}" -background "\"#808080\"" -gravity center -extent "${eq_w}x${resolution}" "${file_eq360}"

    cube_h=$((resolution * 6))
    exec_cmd ffmpeg -i "${file_eq360}" -y -vf "v360=equirect:c1x6:out_forder=${cube_faces[$lens]}:out_frot=${cube_rot[$lens]}:w=${resolution}:h=${cube_h}" -qmin 1 -qscale:v 1 "${file_cube}"
}

for lens in {1..8}; do
    generate_eq180 ${lens}
    generate_cube_map ${lens}
done

cube_length=$((resolution * 6))
lens_src_left=(3 7 1 1 1 5)
lens_src_right=(4 8 2 2 2 6)
for face in {0..5}; do
    y0=$((face * resolution))

    src_left="spl_${input}_${lens_src_left[$face]}_cube.JPG"
    dest_left="spl_${input}_cube_left_${face}.JPG"
    src_right="spl_${input}_${lens_src_right[$face]}_cube.JPG"
    dest_right="spl_${input}_cube_right_${face}.JPG"

    exec_cmd convert "${src_left}" -crop "${resolution}x${resolution}+0+${y0}" "${dest_left}"
    exec_cmd convert "${src_right}" -crop "${resolution}x${resolution}+0+${y0}" "${dest_right}"
done

# fill in the top and bottom sides of the cube.
# we must copy triangles from the other lens to create a full frame
hres=$((resolution / 2))
fres=${resolution}

# one group per lens per up/down. (ie lens * cube_sides)
lens_crop=( "${hres}x${fres}+0+$((2 * fres))" \
                "${fres}x${hres}+0+$((2 * fres))" \
                "${hres}x${fres}+${hres}+$((2 * fres))" \
                "${hres}x${fres}+0+$((3 * fres))" \
                "${fres}x${hres}+0+$((3 * fres + hres))" \
                "${hres}x${fres}+${hres}+$((3 * fres))" )

lens_draw=("\"polygon ${hres},${hres} 0,${fres} 0,0\"" \
               "\"polygon ${hres},${hres} ${fres},0 0,0\"" \
               "\"polygon ${hres},${hres} ${fres},0 ${fres},${fres}\"" \
               "\"polygon ${hres},${hres} 0,${fres} 0,0\"" \
               "\"polygon ${hres},${hres} ${fres},${fres} 0,${fres}\"" \
               "\"polygon ${hres},${hres} ${fres},0 ${fres},${fres}\"")

cube_side=(2 2 2 3 3 3)
lenses=(7 5 3 7 5 3)
eyes=("left" "right")

for e in {0..1}; do
    eye="${eyes[$e]}"
    for r in {0..5}; do
        s="${cube_side[$r]}"
        crop="${lens_crop[$r]}"
        draw="${lens_draw[$r]}"
        lens=$(( lenses[r] + e ))

        exec_cmd convert "spl_${input}_${lens}_cube.JPG" -crop "${crop}" +repage "spl_${input}_cube_${eye}_${s}_tmp.JPG"
        exec_cmd convert "spl_${input}_cube_${eye}_${s}.JPG" -fill "spl_${input}_cube_${eye}_${s}_tmp.JPG" -draw "${draw}" "spl_${input}_cube_${eye}_${s}_app.JPG"
        mv "spl_${input}_cube_${eye}_${s}_app.JPG" "spl_${input}_cube_${eye}_${s}.JPG"
        rm "spl_${input}_cube_${eye}_${s}_tmp.JPG"
    done
done


exec_cmd montage "spl_${input}_cube_left_*.JPG" -tile 1x6 -geometry +0+0 "spl_${input}_cube_left.JPG"
exec_cmd montage "spl_${input}_cube_right_*.JPG" -tile 1x6 -geometry +0+0 "spl_${input}_cube_right.JPG"

exec_cmd ffmpeg -i "spl_${input}_cube_left.JPG" -y -vf "v360=c1x6:equirect" -qmin 1 -qscale:v 1 "spl_${input}_eq360_left.JPG"
exec_cmd ffmpeg -i "spl_${input}_cube_right.JPG" -y -vf "v360=c1x6:equirect" -qmin 1 -qscale:v 1 "spl_${input}_eq360_right.JPG"

exec_cmd montage "spl_${input}_eq360_left.JPG" "spl_${input}_eq360_right.JPG" -tile 1x2 -geometry +0+0 "${output}.JPG"



if (( clean == 1 )); then
    r=`rm spl_${input}_*.JPG 2>&1`
    r=`rm spl_${input}_*.JPG 2>&1`
fi


exit 0
