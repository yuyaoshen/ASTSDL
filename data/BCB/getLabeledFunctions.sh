#!/bin/bash

path=.
src_path=functions_Reduced_astens_antlr

rm $path/dataset
mkdir $path/dataset

cat labeledSamples.txt | while read line
do
	filename=$(echo $line | awk '{print $1}')
	label=$(echo $line | awk '{print $2}')
	cp $path/$src_path/$filename".java.txt" $path/dataset/$label"@"$filename".java.txt"
done
