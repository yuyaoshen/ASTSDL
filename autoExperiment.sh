#!/bin/bash

path=.
rm $path/Samples*

round=5
for((i=1;i<=round;i++))
do
	python splitData.py
	python trainingModel.py
done
rm $path/Samples*
