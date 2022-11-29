#!/bin/bash
for num_parameters in {25,400}
# for num_parameters in {25,50}
do
	for i in {1..30}
	do
		# Standard version, no mutation, no stopping small population
		mkdir -p oneMax${num_parameters}_result/mixedAlgo/result$i &
		./MO_GOMEA 0 2 $num_parameters 1000 10000000 1000000 $i 0 &
		# With weak mutation
		#./MO_GOMEA -m 0 2 $num_parameters 1000 10000000 1000000
		# With strong mutation
		#./MO_GOMEA -M 0 2 $num_parameters 1000 10000000 1000000
		# With stopping small populations
		#./MO_GOMEA -z 0 2 $num_parameters 1000 10000000 1000000
	done
	wait
	echo $num_parameters | awk '{printf("%d",$1);}' >> onemax_statistics.dat
	cat "number_of_evaluations_when_all_points_found_${num_parameters}.dat" | awk 'BEGIN{total_eval=0.0;num=0;std=0.0;}{total_eval += $1;num++;eval[num]=$1;}END{avg=total_eval/num; for(j=1;j<=num;j++){std+=(eval[j]-avg)*(eval[j]-avg);} std=sqrt(std/num); printf(" %f %f\n", avg, std);}' >> onemax_statistics.dat
done

