#!/bin/zsh
if (($# != 1)) ; then
	echo "Usage:$0 dir"
	exit
fi
dir=$1
for filename in "$dir"/*.txt; do
	KVAL1="$(cut -d , -f1 <<< $filename)"
	KVAL="$(cut -d '(' -f2 <<< $KVAL1)"
	echo python RNN.py --data-dir="$filename" --kval="$KVAL"
	python RNN.py --data-dir="$filename" --kval="$KVAL"
done
