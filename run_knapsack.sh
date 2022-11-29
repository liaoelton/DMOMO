#!/bin/bash
for i in {1..30}
do
	echo ./MO_GOMEA -r 2 2 250 2000 10000000 1000000 $i &
	mkdir -p knapsack250_result/onlyFI_ims/result$i &
	# Standard version, no mutation, no stopping small population
	./MO_GOMEA -r 2 2 250 2000 20000000 1000000 $i &
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


