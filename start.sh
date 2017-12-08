#!/bin/zsh
if (($# != 2)) ; then
        echo "Usage:$0 traindir predictdir"
        exit
fi
traindir=$1
predictdir=$2

./train.sh "$traindir"
./predict.sh "$predictdir"
