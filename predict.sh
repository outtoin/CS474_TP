#!/bin/zsh
for dirname in checkpoints/*; do
	FILENAME="$(cut -d '-' -f2 <<< $dirname)"
	KVAL1="$(cut -d , -f1 <<< $FILENAME)"
	KVAL="$(cut -d '(' -f2 <<< $KVAL1)"
	echo python predict.py --checkpoint="$dirname" --test-dir=test/"$FILENAME" --kval="$KVAL"
	python predict.py --checkpoint="$dirname" --test-dir=test/"$FILENAME" --kval="$KVAL"
done

