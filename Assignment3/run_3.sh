#! /bin/bash

row_plan=(4000 5000)
col_plan=(100 800 1000 2000 3000 4000 5000)
density_plan=(0.25 0.5 0.75 1.0)
echo "Starts at: $(date)"

for row in ${row_plan[@]};
do
    for col in ${col_plan[@]};
    do
        for d in ${density_plan[@]};
        do
            for n in {1..10..1};
            do
                prun -np 1 -reserve $1 ./coo_spmv -r -m ${row} -n ${col} -d ${d} -o tmp
            done
        done
    done
done