#!/bin/zsh
for filename in data/*.txt; do
	KVAL1="$(cut -d , -f1 <<< $filename)"
	KVAL="$(cut -d '(' -f2 <<< $KVAL1)"
	echo python RNN.py --data-dir="$filename" --kval="$KVAL"
	python RNN.py --data-dir="$filename" --kval="$KVAL"
done
