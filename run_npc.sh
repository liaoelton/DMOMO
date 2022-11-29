#!/bin/bash
# for num_parameters in {6,12,25,50,100}
EXP_NAME=onlyFIConditionalNotStop
echo $EXP_NAME
for num_parameters in {6,25,50,100,200,400}
do
	if [ $num_parameters -eq 6 ]; then
		limit=10000
		interval=1000
	fi
	if [ $num_parameters -eq 12 ]; then
		limit=50000
		interval=5000
	fi
	if [ $num_parameters -eq 25 ]; then
		limit=500000
		interval=50000
	fi
	if [ $num_parameters -eq 50 ]; then
		limit=5000000
		interval=500000
	fi
	if [ $num_parameters -eq 100 ]; then
		limit=20000000
		interval=2000000
	fi
	if [ $num_parameters -eq 200 ]; then
		limit=20000000
		interval=2000000
	fi
	if [ $num_parameters -eq 400 ]; then
		limit=20000000
		interval=2000000
	fi
	for i in {1..50}
	do
		echo ./MO_GOMEA 4 2 $num_parameters 1000 $limit $interval $i ${EXP_NAME} &
		# Standard version, no mutation, no stopping small population
		mkdir -p maxcut${num_parameters}_result/${EXP_NAME}/result$i &
		./MO_GOMEA 4 2 $num_parameters 1000 $limit $interval $i 0 ${EXP_NAME} | tee maxcut${num_parameters}_result/${EXP_NAME}/result$i/log.txt &
		# With weak mutation
		#./MO_GOMEA -m 4 2 $num_parameters 1000 $limit $interval
		# With strong mutation
		#./MO_GOMEA -M 4 2 $num_parameters 1000 $limit $interval
		# With stopping small populations
		#./MO_GOMEA -z 4 2 $num_parameters 1000 $limit $interval
		# for f in elitist*.dat;
		# 	do mv $f ${num_parameters}_${i}_${f};
    done
	wait
	echo All $num_parameters maxcut tasks finished
done

#!/bin/bash

for num_parameters in {100,250,500,750}
do
    if [ $num_parameters -eq 100 ]; then
        limit=10000000
        interval=500000
    fi
    if [ $num_parameters -eq 250 ]; then
        limit=10000000
        interval=1000000
    fi
    if [ $num_parameters -eq 500 ]; then
        limit=15000000
        interval=1500000
    fi
    if [ $num_parameters -eq 750 ]; then
        limit=20000000
        interval=2000000
    fi
    for i in {1..30}
    do
        echo ./MO_GOMEA -r 2 2 $num_parameters 2000 $limit $interval $i &
		mkdir -p maxcut${num_parameters}_result/${EXP_NAME}/result$i &
        mkdir -p knapsack${num_parameters}_result/${EXP_NAME}/result$i &
        # Standard version, no mutation, no stopping small population
        ./MO_GOMEA -r 2 2 $num_parameters 2000 $limit $interval $i 0 &
        
        # With weak mutation
        #./MO_GOMEA -r -m 2 2 $num_parameters 2000 $limit $interval
        # With strong mutation
        #./MO_GOMEA -r -M 2 2 $num_parameters 2000 $limit $interval
        # With stopping small populations
        #./MO_GOMEA -r -z 2 2 $num_parameters 2000 $limit $interval
        # for f in elitist*.dat;
        # 	do mv $f ${num_parameters}_${i}_${f};
        # done
    done
    wait
    echo All $num_parameters knapsack tasks finished
done
