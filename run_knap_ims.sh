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
        mkdir -p knapsack${num_parameters}_result/onlyFIConditionalNotStop2/result$i &
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
    echo All $num_parameters tasks finished
done
