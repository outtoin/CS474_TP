#!/bin/zsh
if (($# != 2)) ; then
        echo "Usage:$0 checkpoint_dir test_dir"
        exit
fi
CHECKPOINTDIR=$1
TESTDIR=$2
for dirname in "$CHECKPOINTDIR"/*; do
	FILENAME="$(cut -d '-' -f2 <<< $dirname)"
	KVAL1="$(cut -d , -f1 <<< $FILENAME)"
	KVAL="$(cut -d '(' -f2 <<< $KVAL1)"
	echo python ensemble.py --checkpoint="$dirname" --test-dir="$TESTDIR"/"$FILENAME" --kval="$KVAL"
	python ensemble.py --checkpoint="$dirname" --test-dir="$TESTDIR"/"$FILENAME" --kval="$KVAL"
done

