#! /bin/bash
make clean
make

N_EXP=30

# Count the matrix
N_MTX=$(ls ../data | wc -l)
APP_NAME=${PWD##*/}
echo "===${APP_NAME}==="
echo "${N_MTX} matrix to measure"
for d in $(find "../data" -maxdepth 1 -type f)
do 
    MTX=${d##*/}
    MTX=${MTX%%.*}
    # prun -np 1 -v ./transpose ${d} tmp.mtx
    ./transpose ${d} tmp.mtx
    for ((i=1;i<=N_EXP;i++));
    do
        TIME=$(./matmul $d tmp.mtx result.mtx)
        echo $MTX,$1,$TIME
    done

    rm tmp.mtx result.mtx
done


