#! /bin/bash

N_EXP=10

# Count the matrix
N_MTX=$(ls data | wc -l)
APP_NAME=${PWD##*/}
echo "===${APP_NAME}==="
echo "${N_MTX} matrix to measure"
for d in $(find "data" -maxdepth 1 -type f)
do 
    MTX=${d##*/}
    MTX=${MTX%%.*}
    for ((i=1;i<=N_EXP;i++));
    do
        TIME=$(prun -np 1 -reserve $1 ./coo_spmv -i $d -o tmp)
        echo $MTX,$TIME
        TIME=$(prun -np 1 -reserve $1 ./csr_spmv -i $d -o tmp)
        echo $MTX,$TIME
    done
done


