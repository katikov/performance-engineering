
Times=3
reservationId=3589233

for ((i=1;i<="$Times";i++))
do
    # prun -np 1 -reserve "$reservationId"  ./matmul ash958.mtx t_ash958.mtx >> res.txt
    ./matmul ash958.mtx t_ash958.mtx >> res.txt
done

