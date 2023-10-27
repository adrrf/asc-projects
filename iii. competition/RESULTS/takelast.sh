#!/bin/bash
for file in *.out
do
    head -n 100 $file > ${file/all/last}
done

