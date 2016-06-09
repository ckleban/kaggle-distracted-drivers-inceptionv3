#!/bin/bash -e

# To run, this assumes that you are in the directory with the images
# unpacked into train/ and test/.  To run, it works best to use GNU
# parallel as
#
#   ls train/*.jpeg test/*.jpeg | parallel ./prep_image.sh
#
# Otherwise, it also works to do a bash for loop, but this is slower.
#
#   for f in `ls train/*.jpeg test/*.jpeg`; do ./prep_image.sh $f; done
#

size=640x480


#out_dir=processed/run-normal
out_dir=processed
dir_name=`dirname $1`
base_name=`basename $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name

prefix=PNOR1
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
convert -fuzz 10% -trim +repage -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-stretch
out_dir=processed
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PSTR1
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -transparent black -contrast-stretch 2x2% -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-hue-1
out_dir=processed
dir_name=`dirname $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PHUE1
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -modulate 100,100,80 -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-hue-2
out_dir=processed
dir_name=`dirname $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PHUE2
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -modulate 100,100,120 -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-sat-1
out_dir=processed
dir_name=`dirname $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PSAT1
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -modulate 100,80,100 -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-sat-2
out_dir=processed
dir_name=`dirname $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PSAT2
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -modulate 100,120,100 -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-contrast-1
out_dir=processed
dir_name=`dirname $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PCON1
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -sigmoidal-contrast 5,75% -resize $size -gravity center -background black -extent $size -equalize $1 $out

out_dir=processed/run-contrast-2
out_dir=processed
dir_name=`dirname $1`
mkdir -p $out_dir/$dir_name
mkdir -p $out_dir/$dir_name
prefix=PCON2
out=$out_dir/$dir_name/$prefix$base_name
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
    convert -fuzz 10% -trim +repage -sigmoidal-contrast 10,50% -resize $size -gravity center -background black -extent $size -equalize $1 $out
