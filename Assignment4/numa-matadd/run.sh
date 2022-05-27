
Times=3
reservationId=3640459

for ((i=1;i<="$Times";i++))
do
    prun -np 1 -reserve $reservationId  likwid-perfctr -C 0-7 -g NUMA ./matadd 8
    
done

