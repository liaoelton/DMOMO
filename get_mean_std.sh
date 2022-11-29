#!/bin/bash
for num_parameters in {25,50,100,200}
# for num_parameters in {100,50}
do
	echo $num_parameters | awk '{printf("MOGOMEA w/o IMS: ",$1);}' >> trap5_statistics4.dat
	cat "nfe_${num_parameters}_MOGOMEAnoIMS.dat" | awk 'BEGIN{total_eval=0.0;num=0;std=0.0;}{total_eval += $1;num++;eval[num]=$1;}END{avg=total_eval/num; for(j=1;j<=num;j++){std+=(eval[j]-avg)*(eval[j]-avg);} std=sqrt(std/num); printf(" %f %f\n", avg, std);}' >> trap5_statistics4.dat
    echo $num_parameters | awk '{printf("only FI w/o IMS: ",$1);}' >> trap5_statistics4.dat
	cat "nfe_${num_parameters}_onlyFInoIMS.dat" | awk 'BEGIN{total_eval=0.0;num=0;std=0.0;}{total_eval += $1;num++;eval[num]=$1;}END{avg=total_eval/num; for(j=1;j<=num;j++){std+=(eval[j]-avg)*(eval[j]-avg);} std=sqrt(std/num); printf(" %f %f\n", avg, std);}' >> trap5_statistics4.dat
    echo $num_parameters | awk '{printf("MOGOMEA w IMS: ",$1);}' >> trap5_statistics4.dat
	cat "nfe_${num_parameters}_MOGOMEAwIMS.dat" | awk 'BEGIN{total_eval=0.0;num=0;std=0.0;}{total_eval += $1;num++;eval[num]=$1;}END{avg=total_eval/num; for(j=1;j<=num;j++){std+=(eval[j]-avg)*(eval[j]-avg);} std=sqrt(std/num); printf(" %f %f\n", avg, std);}' >> trap5_statistics4.dat
    echo $num_parameters | awk '{printf("only FI w IMS: ",$1);}' >> trap5_statistics4.dat
	cat "nfe_${num_parameters}_onlyFIwIMS.dat" | awk 'BEGIN{total_eval=0.0;num=0;std=0.0;}{total_eval += $1;num++;eval[num]=$1;}END{avg=total_eval/num; for(j=1;j<=num;j++){std+=(eval[j]-avg)*(eval[j]-avg);} std=sqrt(std/num); printf(" %f %f\n", avg, std);}' >> trap5_statistics4.dat
done

