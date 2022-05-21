#! /bin/bash
make clean
make
row_plan=(100 400 800 1000 2000 3000 4000 5000)
col_plan=(100 1000 5000)
density_plan=(0.25 0.5 0.75 1.0)
echo "Starts at: $(date)"

for row in ${row_plan[@]};
do
    for col in ${col_plan[@]};
    do
        for d in ${density_plan[@]};
        do
            for n in {1..3..1};
            do
                prun -np 1 -reserve $2 ./matmul ${row} ${col} ${row} ${d} 32
                # ./matmul ${row} ${col} ${row} ${d} 32
            done
        done
    done
done