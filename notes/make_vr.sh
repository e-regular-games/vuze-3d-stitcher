#!/bin/bash

# requires imagemagick: convert, identify
# requires exiftool
# requires base64

# xmp template which will be used for setting the GImage and GPano properties.
# the template must be used to get the properties in a way that Google Photos will
# think they are correct. Using the default placement with exiftool results in a file
# Google believes is invalid.
xmp_b64="PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPD94cGFja2V0IGJlZ2luPSfvu78nIGlkPSdXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQnPz4KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iSW1hZ2U6OkV4aWZUb29sIDEyLjE2Ij4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiB4bWxuczpHSW1hZ2U9Imh0dHA6Ly9ucy5nb29nbGUuY29tL3Bob3Rvcy8xLjAvaW1hZ2UvIiByZGY6YWJvdXQ9IiI+CiAgICAgICAgIDxHSW1hZ2U6RGF0YT5fX0JBU0U2NERBVEFfXzwvR0ltYWdlOkRhdGE+CiAgICAgICAgIDxHSW1hZ2U6TWltZT5pbWFnZS9qcGVnPC9HSW1hZ2U6TWltZT4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24geG1sbnM6R1Bhbm89Imh0dHA6Ly9ucy5nb29nbGUuY29tL3Bob3Rvcy8xLjAvcGFub3JhbWEvIiByZGY6YWJvdXQ9IiI+CiAgICAgICAgIDxHUGFubzpDcm9wcGVkQXJlYUltYWdlSGVpZ2h0UGl4ZWxzPl9fSEVJR0hURVlFX188L0dQYW5vOkNyb3BwZWRBcmVhSW1hZ2VIZWlnaHRQaXhlbHM+CiAgICAgICAgIDxHUGFubzpJbml0aWFsVmlld0hlYWRpbmdEZWdyZWVzPjE4MDwvR1Bhbm86SW5pdGlhbFZpZXdIZWFkaW5nRGVncmVlcz4KICAgICAgICAgPEdQYW5vOkluaXRpYWxIb3Jpem9udGFsRk9WRGVncmVlcz4xNTA8L0dQYW5vOkluaXRpYWxIb3Jpem9udGFsRk9WRGVncmVlcz4KICAgICAgICAgPEdQYW5vOkNyb3BwZWRBcmVhSW1hZ2VXaWR0aFBpeGVscz5fX1dJRFRIX188L0dQYW5vOkNyb3BwZWRBcmVhSW1hZ2VXaWR0aFBpeGVscz4KICAgICAgICAgPEdQYW5vOkNyb3BwZWRBcmVhTGVmdFBpeGVscz4wPC9HUGFubzpDcm9wcGVkQXJlYUxlZnRQaXhlbHM+CiAgICAgICAgIDxHUGFubzpDcm9wcGVkQXJlYVRvcFBpeGVscz4wPC9HUGFubzpDcm9wcGVkQXJlYVRvcFBpeGVscz4KICAgICAgICAgPEdQYW5vOkZ1bGxQYW5vSGVpZ2h0UGl4ZWxzPl9fSEVJR0hURVlFX188L0dQYW5vOkZ1bGxQYW5vSGVpZ2h0UGl4ZWxzPgogICAgICAgICA8R1Bhbm86RnVsbFBhbm9XaWR0aFBpeGVscz5fX1dJRFRIX188L0dQYW5vOkZ1bGxQYW5vV2lkdGhQaXhlbHM+CiAgICAgICAgIDxHUGFubzpQcm9qZWN0aW9uVHlwZT5lcXVpcmVjdGFuZ3VsYXI8L0dQYW5vOlByb2plY3Rpb25UeXBlPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KPD94cGFja2V0IGVuZD0ndyc/Pgo="

# usage: --write-template tmpl.out
if [ "$1" = "--write-template" ]; then
    if [ -z "$2" ]; then
        echo "usage: --write-template <file>"
        exit 1
    fi
    echo ${xmp_b64} | base64 -d > $2
    exit 0
fi

# param(string) filename with path and extension of the image to format
format_image() {
    file=$1
    if [ ! -f "$file" ]; then
        echo "does not exist: $file"
        return 1
    fi

    file_ext="${file##*.}"
    file_name="${file%.*}"
    file_left="${file_name}_left.${file_ext}"
    file_right="${file_name}_right.${file_ext}"
    file_final="${file_name}_gimage.${file_ext}"

    dims=`identify $file | awk '{ print $3 }'`
    width=`echo $dims | awk -F 'x' '{ print $1 }'`
    height=`echo $dims | awk -F 'x' '{ print $2 }'`
    height_eye=$((height / 2))

    echo "processing: ${file}"
    echo -e "\twidth: ${width}"
    echo -e "\theight: ${height}"
    echo -e "\theight (per eye): ${height_eye}"
    echo -e "\tassuming stacked stereoscopic format"

    convert "$file" -crop "${width}x${height_eye}+0+0" +repage "${file_left}"
    convert "$file" -crop "${width}x${height_eye}+0+${height_eye}" +repage "${file_right}"

    res=`exiftool -overwrite_original -FullPanoHeightPixels=${height_eye} -CroppedAreaImageHeightPixels=${height_eye} "${file_right}"`
    encoded_right=`base64 -w 0 ${file_right}`

    xmp=`echo ${xmp_b64} | base64 -d`
    xmp="${xmp//__WIDTH__/"$width"}"
    xmp="${xmp//__HEIGHTEYE__/"$height_eye"}"
    xmp="${xmp/__BASE64DATA__/"$encoded_right"}"

    echo $xmp > "${file_name}.xmp"
    cp "${file_left}" "${file_final}"
    res=`exiftool -overwrite_original -tagsfromfile "${file_name}.xmp" '-xmp:all<all' "${file_final}"`

    #rm "${file_name}.xmp"
    #rm "${file_right}"
    #rm "${file_left}"

    echo -e "\tcomplete: ${file}"
}

for f in "$@"
do
    format_image "$f"
done
