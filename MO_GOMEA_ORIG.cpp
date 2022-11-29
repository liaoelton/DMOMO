/**
 * MO_GOMEA.c
 *
 * IN NO EVENT WILL THE AUTHORS OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * Multi-Objective Gene-pool Optimal Mixing Evolutionary Algorithm with IMS
 *
 * In this implementation, maximization is assumed.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The software has been constructed based on
 * Linkage Tree Genetic Algorithm (LTGA) and
 * Multi-objective Adapted Maximum-Likelihood Gaussian Model Iterated Density 
 * Estimation Evolutionary Algorithm (MAMaLGaM)
 *
 * Interested readers can refer to the following publications for more details:
 *
 * 1. N.H. Luong, H. La Poutré, and P.A.N. Bosman: Multi-objective Gene-pool 
 * Optimal Mixing Evolutionary Algorithms with the Interleaved Multi-start Scheme. 
 * In Swarm and Evolutionary Computation, vol. 40, June 2018, pages 238-254, 
 * Elsevier, 2018.
 * 
 * 2. N.H. Luong, H. La Poutré, and P.A.N. Bosman: Multi-objective Gene-pool 
 * Optimal Mixing Evolutionary Algorithms. In Dirk V. Arnold, editor,
 * Proceedings of the Genetic and Evolutionary Computation Conference GECCO 2014: 
 * pages 357-364, ACM Press New York, New York, 2014.
 *
 * 3. P.A.N. Bosman and D. Thierens. More Concise and Robust Linkage Learning by
 * Filtering and Combining Linkage Hierarchies. In C. Blum and E. Alba, editors,
 * Proceedings of the Genetic and Evolutionary Computation Conference -
 * GECCO-2013, pages 359-366, ACM Press, New York, New York, 2013. 
 *
 * 4. P.A.N. Bosman. The anticipated mean shift and cluster registration 
 * in mixture-based EDAs for multi-objective optimization. In M. Pelikan and
 * J. Branke, editors, Proceedings of the Genetic and Evolutionary Computation 
 * GECCO 2010, pages 351-358, ACM Press, New York, New York, 2010.
 * 
 * 5. J.C. Pereira, F.G. Lobo: A Java Implementation of Parameter-less 
 * Evolutionary Algorithms. CoRR abs/1506.08694 (2015)
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <math.h>
#include <set>
#include <string>
using namespace std;
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
/*---------------------------- Utility Functions ---------------------------*/
void *Malloc( long size );
void initializeRandomNumberGenerator();
double randomRealUniform01( void );
int randomInt( int maximum );
// double log2( double x );
int* createRandomOrdering(int size_of_the_set);
double distanceEuclidean( double *x, double *y, int number_of_dimensions );

int *mergeSort( double *array, int array_size );
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q );
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q );
/*-------------------------Interpret Command Line --------------------------*/
void interpretCommandLine( int argc, char **argv );
void parseCommandLine( int argc, char **argv );
void parseOptions( int argc, char **argv, int *index );
void printAllInstalledProblems( void );
void optionError( char **argv, int index );
void parseParameters( int argc, char **argv, int *index );
void printUsage( void );
void checkOptions( void );
void printVerboseOverview( void );
/*--------------- Load Problem Data and Solution Evaluations ---------------*/
void evaluateIndividual(char *solution, double *obj, double *con, int objective_index_of_extreme_cluster);
char *installedProblemName( int index );
int numberOfInstalledProblems( void );

void onemaxLoadProblemData();
void trap5LoadProblemData();
void lotzLoadProblemData();
void onemaxProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
double deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, int k, char is_one );
void trap5ProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
void lotzProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);

void knapsackLoadProblemData();
void ezilaitiniKnapsackProblemData();
void knapsackSolutionRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_contraint, int objective_index_of_extreme_cluster);
void knapsackSolutionSingleObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index);
void knapsackSolutionMultiObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint);
void knapsackProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);

void maxcutLoadProblemData();
void ezilaitiniMaxcutProblemData();
void maxcutReadInstanceFromFile(char *filename, int objective_index);
void maxcutProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster );

double **getDefaultFrontOnemaxZeromax( int *default_front_size );
double **getDefaultFrontTrap5InverseTrap5( int *default_front_size );
double **getDefaultFrontLeadingOneTrailingZero( int *default_front_size );
short haveDPFSMetric( void );
double **getDefaultFront( int *default_front_size );
double computeDPFSMetric( double **default_front, int default_front_size, double **approximation_front, int approximation_front_size );
/*---------------------------- Tracking Progress ---------------------------*/
void writeGenerationalStatistics();
void writeCurrentElitistArchive( char final );
void writeCurrentElitistArchiveFrom( char final );
void writeCurrentHistory();
void writeCurrentAngle(double angle, bool is_FI);
void writeCurrentLevel(double level, bool is_FI);
void writeCurrentChangeBits(int len, bool is_FI, int mask);
void writeCurrentMask(int cluster_index, int linkage_group_index, char *result, char *backup);
void logElitistArchiveAtSpecificPoints();
char checkTerminationCondition();
char checkNumberOfEvaluationsTerminationCondition();
char checkVTRTerminationCondition();
void logNumberOfEvaluationsAtVTR();
/*---------------------------- Elitist Archive -----------------------------*/
char isDominatedByElitistArchive( double *obj, double con, char *is_new_nondominated_point, int *position_of_existed_member );
short sameObjectiveBox( double *objective_values_a, double *objective_values_b );
int hammingDistanceInParameterSpace(char *solution_1, char *solution_2);
int hammingDistanceToNearestNeighborInParameterSpace(char *solution, int replacement_position);
void updateElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, int solution_from, int chromosome_index );
void updateElitistArchiveWithReplacementOfExistedMember( char *solution, double *solution_objective_values, double solution_constraint_value, char *is_new_nondominated_point, char *is_dominated_by_archive, int solution_from, int chromosome_index, int *from_which);
void removeFromElitistArchive( int *indices, int number_of_indices );
short isInListOfIndices( int index, int *indices, int number_of_indices );
void addToElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, int solution_from, int chromosome_index);
void adaptObjectiveDiscretization( void );
/*-------------------------- Solution Comparision --------------------------*/
char betterFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index );
char equalFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index );
short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
short constraintWeaklyParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
short paretoDominates( double *objective_values_x, double *objective_values_y );
short weaklyParetoDominates( double *objective_values_x, double *objective_values_y );
/*-------------------------- Linkage Tree Learning --------------------------*/
void learnLinkageTree( int cluster_index );
double *estimateParametersForSingleBinaryMarginal(  int cluster_index, int *indices, int number_of_indices, int *factor_size );
int determineNearestNeighbour( int index, double **S_matrix, int mpm_length );
void printLTStructure( int cluster_index );
/*-------------------------------- MO-GOMEA --------------------------------*/
void initialize();
void initializeMemory();
void initializePopulationAndFitnessValues();
void computeObjectiveRanges( void );

void learnLinkageOnCurrentPopulation();
int** clustering(double **objective_values_pool, int pool_size, int number_of_dimensions, 
                    int number_of_clusters, int *pool_cluster_size );
int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select );
void determineExtremeClusters();
void initializeClusters();
void ezilaitiniClusters();

void improveCurrentPopulation( void );
void copyValuesFromDonorToOffspring(char *solution, char *donor, int cluster_index, int linkage_group_index);
void copyValuesFromDonorToOffspringComplement(char *solution, char *donor, int cluster_index, int linkage_group_index);
void copyFromAToB(char *solution_a, double *obj_a, double con_a, char *solution_b, double *obj_b, double *con_b);
void mutateSolution(char *solution, int lt_factor_index, int cluster_index);
void performMultiObjectiveGenepoolOptimalMixing( int cluster_index, char *parent, double *parent_obj, double parent_con, 
                                            char *result, double *obj, double *con, int *from_which, int chromosome_index );
void performSingleObjectiveGenepoolOptimalMixing( int cluster_index, int objective_index, 
                                char *parent, double *parent_obj, double parent_con,
                                char *result, double *obj, double *con, int *from_which, int chromosome_index);

void selectFinalSurvivors();
void freeAuxiliaryPopulations();
/*-------------------------- Parameter-less Scheme -------------------------*/
void initializeMemoryForArrayOfPopulations();
void putInitializedPopulationIntoArray();
void assignPointersToCorrespondingPopulation();
void ezilaitiniArrayOfPopulation();
void ezilaitiniMemoryOfCorrespondingPopulation();
void schedule_runMultiplePop_clusterPop_learnPop_improvePop();
void schedule();

void initializeCommonVariables();
void ezilaitiniCommonVariables();
void loadProblemData();
void ezilaitiniProblemData();
void run();
int main( int argc, char **argv );
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

long total_mixing_times = 0;
long total_FI_times = 0;
long total_FI_improve_times = 0;
long gen_now = 0;
long right_cnt = 0, FI_right_cnt = 0;
long anti_cnt = 0, FI_anti_cnt = 0;
long both_small_cnt = 0;
long total_complement_count = 0;
long directReplaceCount = 0;
long in_cluster_time = 0;
long total_cnt = 0;
double antiProb = 0.0;

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
char    **population,               /* The population containing the solutions. */
        ***array_of_populations,    /* The array containing all populations in the parameter-less scheme. */
        **offspring,                /* Offspring solutions. */
        **elitist_archive,          /* Archive of elitist solutions. */
        **elitist_archive_copy;     /* Copy of the elitist archive. */
int     *elitist_archive_from;
int     *elitist_archive_chromosome_index;
int     *elitist_archive_evolution_due;
int runTimes;

int     problem_index,                          /* The index of the optimization problem. */
        number_of_parameters,                   /* The number of parameters to be optimized. */
        number_of_generations,                  /* The current generation count. */
        *array_of_number_of_generations,        /* The array containing generation counts of all populations in the parameter-less scheme.*/
        generation_base,                        /* The number of iterations that population of size N_i is run before running 1 iteration of population of size N_(i+1). */
        population_size,                        /* The size of each population. */
        *array_of_population_sizes,             /* The array containing population sizes of all populations in the parameter-less scheme. */
        smallest_population_size,               /* The size of the first population. */
        population_id,                          /* The index of the population that is currently operating. */
        offspring_size,                         /* The size of the offspring population. */

        number_of_objectives,                   /* The number of objective functions. */
        elitist_archive_size,                   /* Number of solutions in the elitist archive. */
        elitist_archive_size_target,            /* The lower bound of the targeted size of the elitist archive. */
        elitist_archive_copy_size,              /* Number of solutions in the elitist archive copy. */
        elitist_archive_capacity,               /* Current memory allocation to elitist archive. */
        number_of_mixing_components,            /* The number of components in the mixture distribution. */
        *array_of_number_of_clusters,           /* The array containing the number-of-cluster of each population in the parameter-less scheme. */
        *population_cluster_sizes,              /* The size of each cluster. */
        **population_indices_of_cluster_members,/* The corresponding index in the population of each solution in a cluster. */
        *which_extreme,                         /* The corresponding objective of an extreme-region cluster. */
        
        t_NIS,                          /* The number of subsequent generations without an improvement (no-improvement-stretch). */
        *array_of_t_NIS,                /* The array containing the no-improvement-stretch of each population in the parameter-less scheme. */
        maximum_number_of_populations,  /* The maximum number of populations that can be run (depending on memory budget). */
        number_of_populations,          /* The number of populations that have been initialized. */

        **mpm,                          /* The marginal product model. */
        *mpm_number_of_indices,         /* The number of variables in each factor in the mpm. */
        mpm_length,                     /* The number of factors in the mpm. */

        ***lt,                          /* The linkage tree, one for each cluster. */
        **lt_number_of_indices,         /* The number of variables in each factor in the linkage tree of each cluster. */
        *lt_length;                     /* The number of factors in the linkage tree of each cluster. */
	      
long    number_of_evaluations,            /* The current number of times a function evaluation was performed. */
        log_progress_interval,            /* The interval (in terms of number of evaluations) at which the elitist archive is logged. */
		maximum_number_of_evaluations,    /* The maximum number of evaluations. */
        *array_of_number_of_evaluations_per_population; /* The array containing the number of evaluations used by each population in the parameter-less scheme. */
		
double  **objective_values,                 /* Objective values for population members. */
        ***array_of_objective_values,       /* The array containing objective values of all populations in the parameter-less scheme. */
        *constraint_values,                 /* Constraint values of population members. */
        **array_of_constraint_values,       /* The array containing constraint values of all populations in the parameter-less scheme. */
        
        **objective_values_offspring,       /* Objective values of offspring solutions. */
        *constraint_values_offspring,       /* Constraint values of offspring solutions. */
                 
        **elitist_archive_objective_values,         /* Objective values of solutions stored in elitist archive. */
        **elitist_archive_copy_objective_values,    /* Copy of objective values of solutions stored in elitist archive. */
        *elitist_archive_constraint_values,         /* Constraint values of solutions stored in elitist archive. */
        *elitist_archive_copy_constraint_values,    /* Copy of constraint values of solutions stored in elitist archive. */

        *objective_ranges,                          /* Ranges of objectives observed in the current population. */
        **array_of_objective_ranges,                /* The array containing ranges of objectives observed in each population in the parameter-less scheme. */
        **objective_means_scaled,                   /* The means of the clusters in the objective space, linearly scaled according to the observed ranges. */
        *objective_discretization,                  /* The length of the objective discretization in each dimension (for the elitist archive). */
        vtr,                              /* The value-to-reach (in terms of Inverse Generational Distance). */
        **MI_matrix;                      /* Mutual information between any two variables */

int64_t random_seed,                      /* The seed used for the random-number generator. */
        random_seed_changing;             /* Internally used variable for randomly setting a random seed. */

char    use_pre_mutation,                   /* Whether to use weak mutation. */
        use_pre_adaptive_mutation,          /* Whether to use strong mutation. */
        use_print_progress_to_screen,       /* Whether to print the progress of the optimization to screen. */
        use_repair_mechanism,               /* Whether to use a repair mechanism (provided by users) if the problem is constrained. */
        *optimization,                      /* Maximization or Minimization for each objective. */
        print_verbose_overview,             /* Whether to print a overview of settings (0 = no). */
        use_vtr,                            /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
        objective_discretization_in_effect, /* Whether the objective space is currently being discretized for the elitist archive. */
        elitist_archive_front_changed;      /* Whether the Pareto front formed by the elitist archive is changed in this generation. */
// MAXCUT Problem Variables
int     ***maxcut_edges, 
        *number_of_maxcut_edges;
double  **maxcut_edges_weights;
// Knapsack Problem Variables
double  **profits,
        **weights,
        *capacities,
        *ratio_profit_weight;
int     *item_indices_least_profit_order;
int     **item_indices_least_profit_order_according_to_objective;
/*------------------- Termination of Smaller Populations -------------------*/
char    *array_of_population_statuses;
double  ***array_of_Pareto_front_of_each_population;
int     *array_of_Pareto_front_size_of_each_population;
char    stop_population_when_front_is_covered;
void updateParetoFrontForCurrentPopulation(double **objective_values_pop, double *constraint_values_pop, int pop_size);
void checkWhichSmallerPopulationsNeedToStop();
char checkParetoFrontCover(int pop_index_1, int pop_index_2);
void ezilaitiniArrayOfParetoFronts();
void initializeArrayOfParetoFronts();
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

void performCheatingOM( int cluster_index, char *parent, double *parent_obj, double parent_con, char *result, double *obj,  double *con, int *from_which, int chromosome_index);
void performForceImprovement( int cluster_index, char *parent, double *parent_obj, double parent_con, char *result, double *obj,  double *con, int *from_which, int chromosome_index);
vector<vector<int>> history;
vector<int> elitist_num_history;
char* exp_name;
char* output_dir;


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Constants -=-=-=-=-=-=-=-=-=-=-=-=-=-*/
#define FALSE 0
#define TRUE 1

#define NOT_EXTREME_CLUSTER -1

#define MINIMIZATION 1
#define MAXIMIZATION 2

#define ZEROMAX_ONEMAX 0
#define TRAP5 1
#define KNAPSACK 2
#define LOTZ 3
#define MAXCUT 4

#define INITIALIZE 0
#define EXTREME_GOM 1
#define MIDDLE_GOM 2
#define EXTREME_FI 3
#define MIDDLE_FI 4
#define CHEATING_OM 5

#define PI 3.14159265
/*-=-=-=-=-=-=-=-=-=-=-=-= Section Utility Function -=-=-=-=-=-=-=-=-=-=-=-*/
double getObjLen(double *B_obj, double *A_obj){
    double res = 0.0;
    for(int i = 0; i < number_of_objectives; i++){
        res += (B_obj[i]-A_obj[i])*(B_obj[i]-A_obj[i]);
    }
    return sqrt(res);
}
// objective_values[donor_index][0], objective_values[donor_index][1], obj_backup[0], obj_backup[1], obj[0], obj[1]
double getAngle(double *donor_obj, double *receiver_obj, double *offspring_obj){
    double result;
    double dot=0.0;
    double desire_len = getObjLen(donor_obj, receiver_obj), actual_len = getObjLen(offspring_obj, receiver_obj);
    for(int i = 0; i < number_of_objectives; i++){
        dot += (donor_obj[i]-receiver_obj[i])*(offspring_obj[i]-receiver_obj[i]);
    }
    dot /= (desire_len*actual_len);
    result = acos (dot) * 180.0 / PI;
    return result;
}

bool checkInCluster( int cluster_index, int linkage_group_index, char *donor ) {
    /*
        Return true if the genotype of about-to-mix masks is in cluster
        Else false
    */
    set<string> masksSet;
    for(int _j = 0; _j < population_cluster_sizes[cluster_index]; _j++) {
        string tmp;
        char *ch = population[population_indices_of_cluster_members[cluster_index][_j]];
        for (int _i = 0; _i < lt_number_of_indices[cluster_index][linkage_group_index]; _i++) {
            int parameter_index = lt[cluster_index][linkage_group_index][_i];
            tmp += ch[parameter_index];
        }
        masksSet.insert(tmp);
    }
    string mask;
    for (int _i = 0; _i < lt_number_of_indices[cluster_index][linkage_group_index]; _i++) {
        int parameter_index = lt[cluster_index][linkage_group_index][_i];
        mask += donor[parameter_index];
    }
    auto it = masksSet.find(mask);  
    if ( it == masksSet.end() ) {  
        // not found  
        return false;  
    }   
    else {  
        // found  
        return true;  
    }  

}

void performForceImprovement( int cluster_index, char *parent, double *parent_obj, double parent_con, char *result, double *obj,  double *con, int *from_which, int chromosome_index, bool doMOGOM)
{
    total_mixing_times++;
    char   *backup, *donor, is_unchanged, changed, is_improved, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, index, donor_index, position_of_existed_member, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;
    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, result, obj, con);
    
    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup);

    number_of_linkage_sets = lt_length[cluster_index] - 1; /* Remove root from the linkage tree. */

    /* Forced Improvement */
    // if (  (!changed) || (t_NIS > (1+floor(log10(population_size))))  ) {
        total_FI_times++;
        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++) {

            /* this is to multi donors trial */
            // for(int donor_i = 0; donor_i < 5; donor_i++) {
            /* this is to multi donors trial */
            
            donor_index = randomInt(elitist_archive_size);
            donor = elitist_archive[donor_index];

            linkage_group_index = order[i];

            /* ORIG MIXING */

            copyValuesFromDonorToOffspring(result, elitist_archive[donor_index], cluster_index, linkage_group_index);

            /* SIMILARITY CHECK */

            // int dist = hammingDistanceInParameterSpace(result, donor);
            // if(dist < number_of_parameters/2) {
            //     copyValuesFromDonorToOffspring(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
            // }
            // else {
            //     copyValuesFromDonorToOffspringComplement(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
            // }

            /* RANDOM MUTATE MIXING */

            // for (int _i = 0; _i < lt_number_of_indices[cluster_index][linkage_group_index]; _i++)
            // {   
            //     int parameter_index = lt[cluster_index][linkage_group_index][_i];
            //     if(randomRealUniform01() > 0.5){
            //         result[parameter_index] = 0;
            //     }
            //     else{
            //         result[parameter_index] = 1;
            //     }
            // }


            /* ANTI MUTATION */

            // if(randomRealUniform01() < antiProb) {
            //     int parameter_index;
            //     for (int _i = 0; _i < lt_number_of_indices[cluster_index][linkage_group_index]; _i++)
            //     {
            //         parameter_index = lt[cluster_index][linkage_group_index][_i];
            //         int zero_cnt = 0;
            //         for(int _j = 0; _j < elitist_archive_size; _j++ ){
            //             if(elitist_archive[_j][parameter_index] == 0)
            //                 zero_cnt++;
            //         }
            //         double zero_prob = double(zero_cnt)/double(elitist_archive_size);
            //         if(randomRealUniform01() > zero_prob){
            //             result[parameter_index] = 0;
            //         }
            //         else{
            //             result[parameter_index] = 1;
            //         }
            //     }   
            // }
            // else{
            //     copyValuesFromDonorToOffspring(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
            // }
            



            mutateSolution(result, linkage_group_index, cluster_index);

            int mask_len = lt_number_of_indices[cluster_index][linkage_group_index];
            
            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            int change_bits_cnt = 0;
            for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ ) {
                if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] ) {
                    is_unchanged = FALSE;
                    change_bits_cnt ++;
                    // break;
                }
            }

            if( is_unchanged == FALSE ) {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, MIDDLE_FI, chromosome_index, from_which);

                if (is_new_nondominated_point){
                    // printf("change bits count: %d/%d\n", change_bits_cnt, mask_len);
                    writeCurrentChangeBits(change_bits_cnt, true, mask_len);
                    double angle = getAngle(elitist_archive_objective_values[donor_index], obj_backup, obj);
                    writeCurrentAngle(angle, true);
                }
                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    // writeCurrentMask(cluster_index, linkage_group_index, result,backup);
                    *from_which = 7;
                    total_FI_improve_times++;
                    copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);

                    writeCurrentLevel(i, true);

                    /* Count the times mask genotype not in current cluster */
                    if(!checkInCluster(cluster_index, linkage_group_index, result)){
                        in_cluster_time++;
                    }
                    total_cnt++;
                    // 1103  mark break to test if FI not stop early
                    if (!doMOGOM){
                        break;
                    }
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
            }
            // COMPLEMENT (better)
            // EVERYTIME -> better than anti-mutation at beginnning, almost same in the end
            // when NOT improved -> poor than anti-mutation
            
            // if( true ){
            //     copyValuesFromDonorToOffspringComplement(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
                    
            //     if(!checkInCluster(cluster_index, linkage_group_index, result)){
            //         in_cluster_time++;
            //     }
            //     total_cnt++;

            //     is_unchanged = TRUE;    
            //     change_bits_cnt = 0;

            //     for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ ) {
            //         if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] ) {
            //             is_unchanged = FALSE;
            //             change_bits_cnt ++;
            //         }
            //     }

            //     if( is_unchanged == FALSE ) {
            //         is_improved = FALSE;
            //         is_improved = FALSE;
            //         evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
            //         updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, MIDDLE_FI, chromosome_index, from_which);

            //         if (is_new_nondominated_point){
            //             writeCurrentChangeBits(change_bits_cnt, true, mask_len);
            //             double angle = getAngle(elitist_archive_objective_values[donor_index], obj_backup, obj);
            //             writeCurrentAngle(angle, true);
            //         }
            //         if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
            //             is_improved = TRUE;
            //         if(is_new_nondominated_point)
            //             is_improved = TRUE;
                    
            //         if ( is_improved )
            //         {
            //             total_complement_count++;
            //             changed = TRUE;
            //             writeCurrentMask(cluster_index, linkage_group_index, result,backup);
            //             *from_which = 7;
            //             // total_FI_improve_times++;
            //             copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
            //             break;
            //         }
            //         else
            //             copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
            //     }
            // }


            /* this is to multi donors trial */
            // }
            // if ( is_improved ) break;
            /* this is to multi donors trial */

        }
        free(order);
        if(!changed) {
            donor_index = randomInt( elitist_archive_size );
            copyFromAToB(elitist_archive[donor_index], elitist_archive_objective_values[donor_index], 
                    elitist_archive_constraint_values[donor_index], 
                    result, obj, con);
            directReplaceCount++;
        }
    // }

    free( backup ); 
    free( obj_backup ); 

}


void performCheatingOM( int cluster_index, char *parent, double *parent_obj, double parent_con, char *result, double *obj,  double *con, int *from_which, int chromosome_index)
{
    total_mixing_times++;
    char   *backup, *donor, is_unchanged, changed, is_improved, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, index, donor_index, position_of_existed_member, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;
    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, result, obj, con);
    
    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup);

    number_of_linkage_sets = lt_length[cluster_index] - 1; /* Remove root from the linkage tree. */
    // ! random choose LT masks (???)
    order = createRandomOrdering(15);
   

//     int cheatingMask[20][1][200] = {{{1, 138, 272, 402, 276, 309, 54, 312, 335, 207, 211, 84, 219, 95, 229, 237, 238, 244, 245, 381, 260, 397, 149, 292, 296, 41, 52, 183, 194, 204, 335, 338, 84, 342, 220, 234, 362, 110, 240, 126, 130, 400, 144, 150, 152, 283, 295, 175, 47, 180, 60, 319, 335, 100, 357, 111, 240, 247, 249, 126, 260, 5, 135, 149, 154, 411, 28, 160, 168, 299, 175, 50, 73, 340, 361, 106, 235, 109, 366, 369, 388, 260, 9, 152, 281, 27, 300, 173, 191, 320, 68, 198, 331, 209, 88, 221, 226, 235, 238, 242, 132, 136, 393, 266, 140, 411, 287, 292, 46, 317, 191, 192, 327, 79, 349, 95, 100, 235, 124, 125, 129, 14, 18, 277, 285, 161, 36, 45, 303, 182, 313, 59, 206, 209, 214, 351, 112, 241, 245, 123, 0, 129, 3, 138, 24, 161, 290, 166, 44, 59, 191, 64, 201, 76, 219, 356, 358, 360, 106, 362, 397, 400, 273, 275, 151, 413, 168, 179, 320, 201, 333, 80, 83, 87, 221, 99, 363, 364, 110, 120, 258, 389, 392, 272, 275, 148, 412, 420, 167, 296, 301, 48, 54, 195, 70, 78, 354, 238, 111, 119}},
//  {{131, 140, 398, 400, 146, 23, 283, 29, 177, 305, 187, 324, 231, 104, 108, 372, 375, 253, 381, 127, 130, 5, 137, 9, 395, 16, 19, 277, 151, 411, 176, 307, 51, 189, 194, 339, 221, 98, 374, 250, 132, 9, 150, 22, 157, 29, 163, 166, 48, 49, 308, 310, 60, 68, 198, 340, 91, 358, 117, 377, 256, 258, 262, 270, 272, 19, 20, 23, 160, 161, 325, 70, 328, 205, 209, 91, 220, 97, 237, 369, 387, 389, 6, 137, 140, 272, 404, 277, 410, 158, 291, 183, 64, 70, 200, 336, 108, 370, 122, 123, 4, 391, 280, 162, 165, 166, 174, 180, 196, 335, 350, 351, 352, 356, 361, 106, 365, 116, 122, 381, 9, 140, 272, 27, 32, 419, 300, 320, 208, 213, 351, 98, 100, 356, 233, 365, 241, 244, 246, 124, 257, 385, 389, 136, 397, 277, 155, 420, 53, 55, 322, 80, 213, 225, 228, 235, 366, 110, 116, 375, 138, 400, 148, 416, 290, 35, 165, 42, 174, 306, 307, 187, 191, 338, 210, 86, 216, 221, 228, 104, 388, 4, 401, 25, 154, 177, 320, 74, 332, 214, 217, 228, 356, 357, 231, 360, 106, 237, 121, 125}},
//  {{386, 5, 133, 14, 277, 412, 285, 292, 37, 44, 177, 319, 210, 99, 116, 245, 372, 251, 380, 254, 261, 390, 271, 409, 44, 48, 305, 179, 184, 192, 72, 201, 337, 340, 355, 104, 105, 365, 250, 380, 3, 388, 261, 138, 396, 140, 142, 282, 285, 420, 180, 181, 202, 74, 83, 94, 98, 234, 371, 249, 1, 134, 8, 267, 139, 401, 405, 166, 297, 177, 59, 319, 197, 332, 206, 235, 363, 376, 380, 381, 258, 388, 16, 273, 21, 31, 34, 290, 167, 295, 43, 51, 52, 313, 57, 68, 329, 333, 101, 241, 278, 280, 33, 291, 163, 293, 36, 300, 302, 50, 52, 320, 68, 210, 340, 341, 360, 234, 106, 253, 5, 6, 12, 13, 281, 284, 416, 292, 47, 305, 307, 187, 189, 64, 323, 79, 84, 214, 372, 380, 258, 132, 6, 9, 140, 145, 18, 279, 151, 281, 56, 199, 79, 207, 209, 214, 343, 356, 229, 374, 4, 391, 266, 139, 270, 271, 286, 419, 48, 180, 329, 331, 333, 84, 87, 223, 104, 360, 366, 122, 391, 146, 20, 149, 23, 151, 300, 320, 322, 324, 76, 80, 231, 240, 372, 117, 245, 378, 126, 127}},
//  {{7, 138, 267, 141, 404, 27, 158, 292, 307, 55, 81, 85, 341, 216, 94, 354, 98, 356, 361, 244, 5, 137, 162, 291, 308, 57, 320, 328, 334, 84, 341, 213, 344, 345, 92, 228, 368, 120, 251, 124, 384, 267, 139, 269, 145, 147, 167, 171, 176, 307, 195, 324, 197, 208, 210, 343, 222, 102, 105, 109, 128, 257, 385, 0, 140, 17, 20, 410, 32, 34, 300, 57, 320, 77, 355, 104, 117, 250, 381, 383, 384, 13, 15, 272, 159, 415, 292, 49, 54, 57, 192, 200, 203, 81, 341, 352, 359, 246, 374, 248, 2, 393, 142, 271, 274, 21, 413, 418, 420, 182, 190, 322, 329, 336, 209, 81, 89, 219, 96, 354, 131, 389, 393, 138, 143, 400, 403, 149, 29, 413, 295, 300, 314, 64, 72, 347, 220, 229, 234, 122, 138, 397, 16, 149, 152, 24, 29, 39, 45, 52, 182, 186, 61, 329, 209, 215, 87, 220, 352, 241, 1, 268, 153, 28, 40, 298, 302, 306, 54, 58, 316, 318, 78, 339, 342, 100, 232, 122, 379, 383, 384, 265, 137, 20, 281, 411, 161, 35, 293, 319, 69, 327, 200, 80, 84, 346, 231, 107, 240, 378}},
//  {{392, 12, 144, 24, 160, 299, 175, 318, 200, 201, 207, 339, 352, 353, 112, 240, 241, 243, 371, 252, 132, 260, 262, 270, 274, 280, 286, 294, 40, 51, 57, 194, 327, 328, 76, 214, 354, 235, 380, 253, 130, 14, 408, 177, 51, 54, 191, 192, 197, 199, 329, 341, 217, 97, 240, 117, 246, 123, 252, 255, 5, 396, 14, 146, 277, 153, 26, 286, 32, 298, 48, 314, 193, 86, 216, 220, 221, 357, 231, 126, 384, 12, 406, 27, 419, 163, 175, 52, 182, 188, 64, 195, 201, 86, 93, 95, 106, 375, 377, 255, 138, 15, 32, 52, 184, 60, 321, 70, 204, 80, 86, 217, 352, 225, 227, 232, 107, 108, 112, 242, 384, 11, 396, 419, 172, 45, 181, 316, 79, 336, 81, 89, 219, 351, 356, 100, 236, 114, 378, 123, 257, 388, 263, 414, 63, 67, 335, 209, 213, 343, 120, 348, 228, 358, 361, 108, 244, 119, 248, 124, 260, 263, 264, 138, 398, 278, 284, 31, 32, 38, 174, 48, 305, 178, 332, 81, 341, 220, 369, 379, 5, 135, 399, 20, 23, 286, 418, 45, 313, 185, 209, 87, 218, 220, 225, 100, 108, 110, 247, 125}},
//  {{131, 23, 26, 286, 42, 46, 58, 186, 187, 326, 70, 76, 347, 99, 355, 235, 240, 370, 246, 118, 128, 388, 402, 18, 148, 288, 168, 300, 46, 180, 58, 192, 71, 342, 215, 352, 230, 232, 108, 122, 1, 397, 273, 21, 278, 157, 285, 297, 46, 177, 49, 204, 97, 357, 105, 235, 123, 246, 251, 253, 135, 9, 394, 269, 410, 292, 40, 172, 49, 66, 69, 201, 202, 92, 97, 109, 369, 378, 382, 255, 258, 2, 4, 260, 133, 141, 23, 30, 158, 41, 303, 184, 315, 64, 204, 206, 224, 104, 111, 116, 130, 4, 393, 401, 288, 161, 173, 176, 184, 188, 64, 324, 326, 80, 82, 84, 215, 360, 108, 124, 262, 21, 150, 151, 22, 158, 290, 420, 165, 310, 187, 70, 200, 330, 235, 239, 370, 247, 380, 255, 135, 16, 400, 19, 151, 25, 27, 30, 181, 311, 57, 195, 331, 211, 348, 351, 106, 238, 377, 251, 142, 399, 19, 22, 34, 47, 180, 57, 74, 79, 80, 341, 344, 89, 345, 219, 93, 99, 109, 119, 393, 267, 398, 19, 410, 155, 298, 300, 58, 189, 190, 333, 338, 210, 344, 236, 240, 118, 120, 378}},
//  {{388, 5, 4, 8, 14, 145, 17, 24, 285, 416, 165, 45, 179, 313, 85, 216, 349, 365, 373, 122, 385, 130, 389, 13, 406, 283, 323, 326, 328, 203, 208, 83, 340, 87, 89, 100, 103, 361, 363, 123, 260, 136, 266, 400, 20, 412, 34, 40, 297, 45, 303, 312, 60, 332, 78, 335, 220, 113, 380, 124, 5, 265, 145, 147, 158, 290, 40, 305, 185, 193, 325, 200, 330, 203, 208, 337, 83, 84, 245, 374, 260, 390, 137, 11, 140, 24, 159, 289, 37, 309, 185, 200, 328, 340, 342, 360, 240, 241, 120, 380, 2, 399, 284, 30, 33, 290, 181, 187, 70, 210, 345, 90, 219, 350, 98, 232, 235, 107, 250, 379, 138, 396, 398, 156, 159, 296, 176, 196, 76, 211, 212, 85, 87, 93, 224, 123, 236, 379, 380, 381, 1, 386, 261, 7, 141, 276, 281, 154, 296, 301, 48, 196, 70, 201, 81, 233, 234, 368, 253, 254, 3, 392, 400, 274, 148, 158, 163, 292, 43, 300, 173, 180, 320, 64, 202, 100, 232, 243, 120, 380, 1, 8, 403, 405, 161, 49, 177, 50, 308, 188, 191, 199, 88, 348, 353, 238, 368, 117, 248, 254}},
//  {{256, 260, 142, 282, 157, 415, 418, 162, 53, 62, 196, 328, 200, 202, 82, 340, 341, 342, 222, 358, 0, 140, 404, 20, 151, 154, 161, 34, 37, 41, 299, 180, 191, 200, 211, 340, 346, 219, 100, 380, 386, 4, 260, 6, 401, 286, 419, 166, 65, 66, 206, 208, 349, 354, 231, 106, 236, 110, 244, 246, 130, 390, 270, 150, 414, 290, 40, 303, 311, 190, 197, 70, 329, 330, 209, 228, 106, 240, 243, 120, 389, 11, 268, 29, 289, 162, 161, 419, 44, 301, 54, 60, 189, 196, 69, 198, 210, 89, 349, 229, 387, 262, 397, 142, 273, 22, 158, 41, 42, 45, 303, 62, 202, 217, 353, 228, 362, 367, 122, 251, 130, 386, 282, 27, 30, 287, 160, 291, 41, 300, 50, 310, 70, 330, 217, 230, 359, 107, 118, 250, 258, 271, 18, 19, 21, 31, 289, 38, 298, 302, 55, 318, 338, 343, 103, 360, 362, 245, 118, 378, 388, 264, 138, 11, 21, 281, 161, 41, 301, 180, 321, 78, 89, 92, 93, 236, 238, 118, 121, 381, 259, 133, 390, 393, 394, 278, 153, 29, 294, 39, 313, 197, 333, 334, 337, 339, 213, 113, 380, 253}},
//  {{134, 391, 392, 138, 141, 23, 161, 37, 41, 301, 312, 320, 201, 203, 81, 341, 94, 101, 361, 119, 1, 394, 140, 280, 411, 283, 40, 172, 60, 320, 200, 333, 80, 340, 85, 220, 354, 356, 111, 112, 0, 388, 389, 392, 140, 142, 29, 289, 48, 309, 189, 196, 69, 330, 209, 349, 229, 233, 245, 119, 396, 141, 14, 274, 22, 25, 162, 295, 314, 318, 194, 201, 334, 210, 214, 94, 228, 114, 377, 254, 262, 142, 271, 162, 170, 183, 194, 82, 339, 211, 85, 345, 220, 102, 362, 238, 242, 122, 380, 382, 133, 263, 10, 12, 303, 304, 177, 183, 203, 338, 83, 343, 91, 223, 224, 103, 368, 243, 250, 252, 150, 418, 164, 170, 54, 61, 189, 330, 331, 204, 206, 335, 210, 83, 90, 350, 230, 110, 376, 250, 259, 266, 16, 275, 276, 278, 410, 156, 296, 181, 185, 203, 334, 336, 213, 216, 94, 356, 236, 376, 4, 23, 24, 413, 164, 40, 44, 49, 178, 51, 184, 58, 319, 320, 344, 224, 230, 104, 364, 249, 385, 134, 140, 13, 400, 18, 404, 280, 284, 300, 60, 200, 337, 100, 360, 240, 120, 250, 380, 255}},
//  {{130, 138, 158, 38, 295, 306, 323, 198, 81, 83, 89, 218, 348, 221, 98, 358, 238, 118, 376, 378, 3, 401, 404, 149, 159, 163, 43, 303, 51, 318, 190, 203, 340, 223, 103, 363, 107, 239, 123, 383, 3, 132, 133, 20, 28, 32, 292, 172, 301, 52, 312, 192, 72, 73, 94, 352, 363, 240, 244, 252, 260, 389, 392, 140, 268, 279, 160, 33, 32, 40, 301, 180, 185, 60, 320, 325, 80, 234, 240, 120, 256, 6, 264, 266, 50, 186, 59, 61, 66, 198, 206, 346, 228, 106, 366, 115, 116, 246, 119, 126, 384, 4, 7, 264, 10, 17, 164, 44, 62, 195, 324, 197, 201, 204, 80, 211, 84, 344, 353, 364, 7, 11, 405, 27, 414, 287, 169, 301, 47, 307, 187, 61, 192, 67, 327, 215, 360, 107, 372, 127, 386, 137, 140, 397, 16, 17, 157, 177, 312, 57, 314, 330, 208, 337, 217, 121, 229, 357, 120, 377, 385, 265, 147, 407, 282, 303, 180, 185, 65, 324, 205, 85, 217, 345, 100, 232, 365, 244, 245, 125, 140, 16, 24, 26, 27, 160, 42, 300, 50, 180, 60, 200, 351, 225, 360, 240, 112, 252, 120, 380}},
//  {{385, 265, 140, 278, 151, 408, 412, 285, 165, 38, 305, 52, 309, 185, 65, 205, 87, 345, 105, 365, 5, 136, 396, 156, 36, 296, 41, 303, 176, 48, 310, 56, 314, 316, 196, 70, 336, 370, 116, 255, 262, 400, 22, 291, 42, 180, 182, 183, 62, 322, 200, 332, 207, 82, 83, 221, 222, 362, 242, 122, 133, 11, 269, 273, 160, 33, 165, 173, 193, 73, 333, 341, 90, 93, 353, 233, 113, 241, 371, 246, 0, 140, 400, 403, 280, 160, 40, 185, 60, 65, 73, 80, 340, 220, 98, 100, 109, 240, 112, 368, 131, 389, 9, 149, 25, 31, 169, 299, 180, 309, 189, 69, 71, 78, 209, 89, 219, 349, 229, 235, 387, 261, 267, 147, 38, 167, 180, 187, 62, 67, 331, 211, 87, 218, 227, 107, 367, 246, 247, 381, 257, 139, 141, 270, 399, 20, 279, 39, 299, 48, 52, 319, 199, 78, 345, 99, 359, 106, 239, 379, 135, 402, 35, 179, 55, 315, 195, 200, 334, 335, 86, 215, 222, 95, 352, 355, 99, 235, 108, 255, 129, 135, 138, 15, 277, 155, 295, 41, 175, 59, 195, 203, 335, 95, 224, 101, 235, 113, 115, 375}},
//  {{264, 273, 24, 153, 284, 32, 418, 164, 41, 304, 184, 60, 188, 64, 344, 219, 224, 104, 364, 244, 388, 260, 134, 6, 149, 160, 300, 180, 60, 320, 200, 203, 334, 340, 100, 240, 245, 117, 120, 380, 138, 13, 273, 402, 33, 293, 173, 48, 51, 73, 333, 336, 212, 220, 93, 353, 100, 113, 373, 253, 130, 12, 398, 270, 144, 30, 170, 49, 50, 180, 182, 190, 70, 330, 210, 345, 350, 366, 370, 250, 385, 260, 5, 265, 395, 16, 291, 165, 305, 65, 325, 205, 210, 85, 105, 364, 365, 240, 369, 245, 386, 392, 266, 398, 146, 26, 171, 172, 46, 50, 66, 326, 333, 206, 213, 86, 346, 100, 366, 246, 0, 260, 140, 403, 20, 412, 60, 320, 200, 205, 219, 220, 225, 100, 360, 104, 111, 240, 118, 120, 394, 14, 29, 34, 41, 306, 50, 54, 74, 202, 334, 340, 214, 94, 354, 356, 234, 114, 126, 254, 261, 21, 155, 419, 171, 301, 181, 61, 321, 335, 341, 348, 221, 121, 101, 361, 237, 120, 377, 381, 384, 263, 264, 20, 24, 184, 57, 63, 64, 324, 70, 198, 204, 84, 344, 224, 104, 364, 238, 246}},
//  {{129, 389, 9, 269, 143, 150, 29, 289, 34, 291, 169, 49, 51, 309, 189, 69, 329, 109, 368, 120, 259, 139, 19, 279, 153, 159, 179, 319, 192, 70, 199, 330, 79, 339, 219, 376, 359, 370, 248, 379, 385, 5, 265, 146, 280, 25, 44, 45, 185, 325, 326, 205, 345, 92, 225, 98, 105, 365, 109, 245, 2, 262, 269, 282, 31, 162, 42, 182, 209, 82, 84, 342, 344, 222, 100, 102, 362, 237, 242, 382, 0, 260, 140, 20, 160, 165, 45, 60, 320, 193, 340, 218, 220, 95, 100, 229, 360, 240, 124, 380, 257, 261, 17, 280, 409, 157, 297, 176, 177, 57, 317, 77, 337, 339, 217, 97, 357, 237, 127, 255, 261, 7, 267, 396, 147, 404, 409, 27, 287, 416, 167, 47, 307, 327, 87, 216, 227, 367, 247, 124, 388, 8, 268, 399, 273, 148, 20, 28, 288, 171, 48, 308, 315, 68, 328, 73, 208, 338, 88, 348, 391, 266, 271, 31, 33, 291, 165, 171, 180, 311, 191, 192, 327, 331, 76, 211, 91, 231, 371, 251, 266, 141, 21, 281, 31, 161, 41, 46, 181, 313, 61, 201, 81, 83, 212, 341, 346, 221, 361, 121}},
//  {{387, 260, 7, 267, 147, 154, 157, 287, 47, 307, 187, 320, 67, 327, 329, 74, 87, 347, 247, 127, 257, 137, 14, 277, 157, 159, 418, 37, 297, 57, 317, 197, 327, 202, 337, 216, 357, 237, 117, 377, 385, 5, 265, 145, 411, 285, 165, 167, 47, 305, 65, 205, 344, 94, 225, 105, 365, 372, 245, 125, 385, 5, 265, 145, 25, 285, 160, 288, 165, 45, 309, 185, 65, 68, 325, 205, 84, 345, 105, 123, 9, 400, 145, 149, 29, 289, 38, 169, 174, 49, 189, 69, 329, 209, 81, 229, 360, 109, 369, 249, 131, 271, 405, 31, 291, 37, 295, 171, 51, 311, 191, 71, 331, 211, 91, 351, 353, 229, 357, 371, 130, 390, 392, 10, 270, 278, 30, 290, 36, 170, 50, 190, 70, 330, 210, 216, 91, 350, 233, 370, 387, 9, 269, 149, 29, 289, 169, 309, 310, 185, 189, 63, 199, 329, 89, 229, 109, 369, 115, 249, 0, 140, 16, 280, 160, 40, 300, 60, 320, 66, 326, 328, 204, 206, 80, 340, 220, 360, 240, 380, 0, 259, 387, 7, 147, 27, 287, 167, 301, 307, 187, 67, 323, 327, 87, 107, 367, 242, 247, 122}},
//  {{385, 265, 25, 285, 305, 180, 185, 64, 65, 325, 205, 85, 345, 218, 91, 105, 365, 244, 245, 125, 385, 257, 5, 391, 265, 145, 25, 285, 305, 185, 65, 66, 335, 85, 345, 225, 105, 365, 240, 245, 260, 140, 280, 283, 418, 40, 180, 60, 320, 200, 80, 336, 340, 214, 220, 100, 360, 240, 118, 380, 386, 6, 266, 26, 286, 33, 166, 46, 305, 66, 326, 206, 78, 344, 346, 226, 106, 366, 238, 126, 392, 12, 399, 272, 32, 292, 171, 172, 304, 52, 312, 192, 332, 76, 212, 92, 102, 232, 372, 252, 130, 390, 10, 140, 270, 150, 30, 159, 290, 50, 310, 185, 60, 190, 195, 70, 330, 90, 350, 370, 1, 131, 391, 11, 271, 151, 31, 291, 294, 171, 311, 71, 331, 211, 351, 357, 364, 371, 375, 251, 259, 139, 17, 19, 405, 279, 159, 299, 179, 181, 59, 319, 199, 329, 339, 219, 359, 379, 119, 123, 256, 257, 136, 264, 396, 276, 156, 36, 296, 44, 176, 316, 196, 326, 330, 76, 336, 96, 236, 116, 259, 268, 19, 279, 153, 159, 39, 299, 179, 52, 59, 319, 199, 73, 79, 339, 346, 99, 359, 379}},
//  {{137, 397, 398, 277, 281, 157, 37, 177, 54, 57, 317, 197, 77, 337, 217, 97, 363, 237, 117, 377, 387, 7, 267, 147, 154, 27, 287, 167, 47, 307, 182, 187, 60, 327, 87, 347, 107, 367, 247, 382, 398, 18, 278, 158, 416, 38, 298, 178, 318, 68, 198, 78, 338, 222, 98, 358, 361, 238, 118, 378, 0, 260, 6, 20, 280, 160, 40, 300, 180, 60, 320, 200, 91, 220, 100, 237, 240, 120, 380, 382, 395, 15, 275, 155, 289, 35, 295, 302, 55, 315, 195, 206, 335, 215, 95, 355, 101, 235, 375, 255, 388, 8, 140, 268, 148, 28, 288, 168, 43, 48, 308, 68, 328, 88, 348, 222, 228, 100, 368, 248, 132, 392, 272, 152, 32, 288, 292, 172, 52, 192, 199, 72, 332, 212, 220, 352, 353, 232, 112, 372, 385, 129, 5, 265, 145, 149, 285, 165, 168, 45, 305, 65, 325, 85, 225, 105, 365, 245, 373, 125, 128, 388, 8, 268, 396, 28, 288, 291, 168, 48, 188, 68, 69, 328, 331, 88, 348, 108, 368, 248, 135, 395, 270, 15, 275, 35, 295, 175, 53, 55, 311, 318, 195, 75, 335, 95, 355, 235, 375, 255}},
//  {{385, 1, 5, 265, 145, 147, 25, 165, 45, 305, 185, 65, 325, 205, 85, 345, 225, 97, 365, 125, 385, 5, 265, 145, 25, 284, 285, 165, 40, 171, 45, 65, 205, 85, 345, 225, 105, 365, 245, 125, 128, 388, 8, 268, 20, 28, 288, 168, 48, 188, 328, 208, 88, 348, 352, 353, 228, 108, 368, 248, 259, 139, 399, 18, 279, 152, 159, 39, 299, 179, 59, 319, 199, 79, 339, 219, 99, 242, 119, 379, 134, 394, 12, 14, 274, 154, 34, 294, 174, 54, 314, 194, 203, 334, 214, 94, 354, 234, 254, 127, 134, 394, 14, 272, 274, 282, 154, 34, 294, 174, 194, 67, 74, 334, 94, 354, 234, 114, 374, 254, 256, 5, 261, 136, 396, 16, 276, 296, 176, 54, 56, 316, 196, 76, 336, 216, 96, 356, 236, 116, 389, 9, 269, 149, 29, 289, 169, 301, 49, 309, 189, 69, 89, 349, 229, 105, 109, 369, 249, 124, 130, 261, 390, 10, 274, 150, 290, 170, 299, 50, 310, 70, 330, 210, 90, 350, 230, 110, 370, 250, 3, 5, 263, 143, 283, 163, 43, 303, 305, 183, 203, 83, 343, 223, 103, 363, 243, 248, 123, 383}},
//  {{132, 6, 392, 12, 272, 152, 32, 292, 172, 312, 59, 192, 72, 332, 212, 92, 352, 232, 112, 372, 1, 261, 141, 17, 21, 281, 161, 41, 301, 181, 61, 201, 81, 341, 221, 361, 241, 121, 381, 254, 2, 132, 392, 12, 272, 152, 32, 292, 172, 52, 312, 185, 192, 72, 332, 92, 232, 112, 372, 252, 133, 393, 13, 273, 153, 33, 293, 299, 173, 53, 313, 73, 333, 213, 93, 96, 353, 233, 373, 253, 0, 140, 280, 160, 40, 170, 300, 180, 60, 320, 200, 80, 340, 220, 100, 360, 240, 120, 380, 254, 132, 392, 12, 272, 152, 292, 172, 52, 312, 192, 72, 332, 204, 92, 352, 232, 112, 115, 372, 252, 132, 392, 12, 152, 292, 170, 172, 303, 52, 312, 192, 72, 332, 212, 92, 352, 232, 112, 372, 252, 256, 2, 136, 396, 16, 276, 156, 36, 296, 176, 56, 316, 196, 216, 96, 356, 236, 116, 376, 253, 129, 389, 9, 269, 149, 29, 289, 169, 309, 189, 69, 329, 207, 89, 349, 229, 109, 369, 376, 249, 258, 138, 395, 398, 18, 278, 158, 298, 302, 178, 58, 198, 78, 338, 218, 98, 358, 238, 118, 378}},
//  {{135, 395, 15, 275, 35, 295, 175, 55, 315, 195, 75, 335, 215, 95, 355, 235, 115, 244, 375, 255, 258, 138, 398, 18, 278, 162, 38, 298, 178, 58, 318, 198, 78, 338, 218, 98, 358, 238, 118, 378, 384, 4, 264, 144, 24, 284, 162, 164, 44, 304, 184, 64, 204, 84, 344, 224, 104, 364, 244, 124, 386, 387, 7, 267, 27, 287, 167, 47, 307, 187, 67, 327, 207, 87, 347, 227, 107, 367, 247, 127, 1, 261, 21, 281, 161, 41, 301, 181, 61, 321, 201, 81, 341, 221, 101, 357, 361, 241, 121, 381, 131, 391, 11, 271, 146, 31, 291, 171, 51, 311, 191, 71, 331, 211, 91, 351, 231, 111, 371, 251, 387, 7, 267, 147, 27, 287, 300, 47, 307, 187, 67, 327, 207, 87, 347, 227, 107, 367, 247, 127, 385, 131, 11, 271, 151, 31, 291, 171, 51, 311, 191, 71, 331, 211, 91, 351, 231, 111, 371, 251, 133, 393, 13, 273, 153, 33, 293, 173, 53, 313, 57, 193, 333, 213, 93, 353, 233, 113, 373, 253, 128, 388, 8, 268, 148, 28, 168, 48, 308, 310, 188, 68, 328, 208, 88, 348, 228, 108, 368, 248}},
//  {{336, 376, 156, 176, 276, 116, 216, 136, 296, 256, 16, 396, 96, 196, 76, 236, 356, 316, 56, 36, 62, 382, 302, 2, 182, 262, 142, 22, 102, 222, 82, 162, 282, 362, 242, 322, 122, 202, 42, 342, 400, 260, 20, 140, 100, 220, 280, 180, 320, 120, 300, 160, 340, 80, 360, 240, 200, 380, 40, 60, 268, 188, 368, 348, 128, 8, 288, 328, 148, 248, 68, 208, 388, 108, 48, 88, 228, 308, 28, 168, 59, 119, 279, 139, 239, 179, 39, 159, 299, 79, 339, 319, 219, 19, 199, 399, 379, 359, 259, 99, 115, 255, 135, 35, 195, 295, 395, 175, 215, 155, 15, 355, 315, 335, 235, 375, 75, 95, 55, 275, 45, 285, 245, 85, 225, 25, 265, 5, 385, 125, 325, 105, 165, 365, 185, 145, 65, 345, 305, 205, 246, 286, 146, 6, 326, 346, 26, 266, 126, 226, 106, 306, 86, 206, 66, 386, 166, 366, 46, 186, 142, 22, 242, 382, 302, 102, 342, 222, 162, 122, 202, 62, 362, 82, 322, 262, 182, 282, 42, 2, 375, 235, 175, 55, 75, 35, 15, 335, 315, 155, 295, 395, 255, 135, 355, 215, 115, 195, 95, 275}}};


//     int knapCheatMask[50][8] = 
// {{18, 2, 98, 71, 71, 4, 96, 79},
// {99, 2, 71, 35, 8, 79, 4, 56},
// {60, 2, 82, 85, 45, 44, 88, 4},
// {18, 2, 82, 85, 84, 4, 96, 79},
// {36, 28, 46, 69, 4, 96, 79, 80},
// {2, 82, 85, 26, 45, 44, 79, 4},
// {69, 46, 2, 71, 15, 27, 51, 95},
// {57, 82, 2, 67, 8, 44, 27, 52},
// {2, 71, 98, 50, 39, 4, 51, 8},
// {50, 2, 71, 69, 80, 27, 51, 52},
// {15, 2, 82, 71, 4, 51, 8, 45},
// {14, 2, 82, 67, 44, 4, 96, 79},
// {26, 2, 71, 69, 51, 45, 57, 8},
// {2, 85, 67, 49, 8, 4, 96, 79},
// {69, 2, 50, 36, 15, 27, 96, 4},
// {60, 2, 82, 85, 44, 4, 79, 8},
// {26, 2, 71, 31, 45, 51, 4, 8},
// {36, 28, 46, 69, 51, 96, 45, 8},
// {46, 2, 71, 50, 44, 51, 96, 79},
// {98, 85, 82, 71, 44, 4, 96, 79},
// {71, 12, 85, 82, 95, 27, 51, 4},
// {97, 12, 82, 28, 51, 96, 44, 45},
// {15, 12, 82, 71, 51, 96, 80, 8},
// {26, 28, 18, 50, 44, 4, 96, 79},
// {71, 12, 85, 82, 45, 44, 4, 96},
// {35, 28, 18, 79, 79, 8, 44, 45},
// {98, 97, 18, 30, 27, 95, 61, 96},
// {36, 71, 97, 44, 66, 45, 51, 27},
// {50, 18, 28, 69, 27, 51, 45, 44},
// {14, 97, 18, 82, 19, 4, 79, 41},
// {49, 97, 50, 2, 27, 51, 45, 44},
// {56, 18, 28, 30, 44, 45, 51, 66},
// {18, 28, 50, 79, 51, 96, 79, 45},
// {60, 71, 12, 85, 61, 45, 51, 27},
// {44, 97, 71, 98, 45, 44, 4, 79},
// {71, 97, 26, 44, 45, 44, 4, 96},
// {49, 71, 97, 44, 45, 96, 51, 1},
// {97, 71, 79, 44, 27, 95, 88, 4},
// {31, 12, 85, 2, 27, 51, 45, 39},
// {36, 12, 85, 2, 15, 51, 44, 45},
// {30, 85, 12, 2, 51, 27, 45, 44},
// {43, 18, 98, 69, 44, 45, 96, 66},
// {3, 28, 71, 30, 66, 15, 1, 51},
// {82, 12, 85, 30, 8, 57, 27, 95},
// {69, 28, 97, 71, 96, 4, 79, 80},
// {14, 18, 71, 89, 14, 4, 8, 44},
// {85, 12, 82, 30, 96, 8, 44, 27},
// {69, 82, 12, 85, 14, 4, 96, 8},
// {69, 97, 2, 85, 88, 51, 27, 14},
// {12, 82, 30, 98, 44, 27, 45, 51}};

// FOR knap 250
    int knapCheatMask[50][10] = 
{{238, 75, 0, 35, 159, 75, 64, 200, 221, 114},
{166, 35, 220, 159, 75, 64, 54, 125, 89, 139},
{50, 51, 156, 150, 35, 139, 221, 125, 15, 161},
{156, 192, 118, 35, 150, 180, 95, 221, 114, 96},
{249, 59, 75, 35, 159, 89, 125, 221, 239, 54},
{155, 121, 103, 173, 238, 200, 64, 239, 89, 139},
{196, 35, 156, 242, 159, 221, 239, 125, 89, 54},
{154, 159, 156, 192, 35, 96, 125, 89, 239, 139},
{146, 41, 214, 59, 0, 53, 239, 64, 197, 15},
{249, 220, 35, 166, 159, 117, 239, 221, 114, 54},
{51, 75, 156, 192, 35, 96, 125, 139, 237, 217},
{239, 59, 75, 159, 156, 139, 77, 64, 81, 53},
{74, 59, 190, 159, 35, 95, 221, 114, 239, 125},
{90, 190, 159, 160, 75, 28, 53, 114, 197, 89},
{119, 190, 160, 39, 35, 95, 221, 114, 239, 89},
{119, 59, 75, 159, 35, 139, 221, 89, 19, 64},
{0, 35, 220, 159, 166, 237, 243, 89, 70, 114},
{238, 103, 75, 0, 156, 46, 221, 114, 239, 15},
{126, 59, 75, 35, 156, 146, 64, 221, 114, 239},
{220, 35, 166, 0, 195, 57, 77, 139, 64, 114},
{35, 220, 166, 121, 192, 212, 15, 161, 64, 197},
{62, 0, 75, 159, 59, 221, 114, 239, 125, 89},
{35, 156, 192, 75, 242, 212, 54, 221, 96, 77},
{199, 69, 0, 195, 192, 240, 125, 89, 239, 200},
{238, 103, 75, 0, 159, 139, 175, 239, 114, 53},
{199, 159, 190, 59, 35, 197, 15, 139, 161, 112},
{103, 238, 75, 0, 159, 177, 53, 239, 221, 125},
{214, 41, 190, 39, 160, 117, 239, 221, 125, 89},
{120, 166, 35, 156, 242, 15, 139, 221, 114, 239},
{49, 39, 190, 159, 35, 77, 161, 175, 190, 27},
{197, 220, 35, 159, 190, 53, 64, 190, 161, 175},
{47, 197, 220, 35, 156, 155, 53, 139, 161, 217},
{195, 156, 0, 173, 57, 96, 239, 112, 114, 125},
{195, 156, 192, 150, 35, 139, 53, 77, 161, 221},
{126, 59, 75, 159, 0, 117, 190, 77, 114, 54},
{179, 160, 190, 39, 214, 139, 53, 64, 221, 190},
{160, 220, 35, 121, 103, 239, 139, 53, 77, 161},
{181, 159, 75, 35, 156, 243, 125, 114, 96, 117},
{103, 173, 156, 195, 118, 221, 114, 64, 53, 19},
{39, 121, 35, 166, 192, 96, 53, 54, 161, 114},
{143, 220, 156, 150, 159, 5, 15, 243, 114, 117},
{89, 41, 21, 121, 35, 239, 36, 95, 161, 53},
{126, 220, 156, 192, 150, 57, 239, 197, 175, 217},
{105, 35, 166, 220, 192, 77, 19, 190, 117, 5},
{69, 75, 35, 220, 159, 96, 53, 64, 221, 114},
{236, 0, 75, 35, 156, 46, 239, 155, 114, 221},
{41, 21, 241, 35, 75, 28, 53, 64, 221, 190},
{41, 0, 195, 156, 150, 221, 64, 53, 77, 139},
{0, 195, 69, 121, 21, 5, 77, 53, 139, 161},
{113, 160, 190, 39, 214, 190, 64, 53, 197, 175}};

    /* for ell 500 */
    // int knapCheatMask[40][14] = {{309, 64, 485, 486, 404, 294, 468, 228, 138, 193, 118, 26, 342, 340},
    //                             {71, 362, 384, 406, 468, 285, 416, 10, 342, 26, 138, 246, 193, 293},
    //                             {494, 336, 119, 39, 406, 116, 283, 340, 138, 246, 193, 293, 228, 26},
    //                             {19, 485, 486, 404, 294, 200, 468, 87, 342, 26, 118, 193, 293, 271},
    //                             {309, 135, 485, 404, 494, 426, 200, 77, 186, 228, 246, 193, 26, 257},
    //                             {294, 404, 200, 468, 486, 494, 426, 295, 138, 246, 193, 293, 228, 443},
    //                             {434, 404, 486, 494, 426, 469, 154, 369, 228, 293, 246, 193, 42, 118},
    //                             {468, 404, 294, 485, 486, 200, 366, 340, 342, 26, 87, 443, 439, 258},
    //                             {406, 362, 468, 119, 343, 285, 32, 340, 257, 87, 263, 274, 450, 248},
    //                             {283, 202, 34, 135, 326, 452, 76, 26, 271, 342, 27, 293, 138, 42},
    //                             {412, 485, 404, 294, 119, 154, 62, 295, 27, 293, 342, 70, 138, 42},
    //                             {89, 404, 294, 485, 494, 426, 434, 271, 342, 27, 293, 70, 138, 246},
    //                             {146, 404, 119, 154, 343, 434, 362, 454, 26, 271, 27, 293, 138, 42},
    //                             {39, 485, 404, 486, 494, 426, 336, 70, 27, 293, 138, 42, 342, 228},
    //                             {336, 496, 362, 294, 237, 222, 146, 26, 271, 342, 27, 146, 257, 138},
    //                             {452, 154, 343, 119, 496, 485, 362, 257, 27, 342, 293, 138, 228, 283},
    //                             {469, 119, 294, 222, 362, 406, 211, 342, 457, 453, 26, 70, 252, 263},
    //                             {468, 119, 294, 496, 154, 146, 434, 228, 138, 70, 399, 342, 487, 498},
    //                             {313, 343, 496, 119, 294, 468, 362, 146, 457, 26, 293, 246, 70, 228},
    //                             {421, 434, 222, 119, 146, 496, 362, 369, 27, 70, 118, 110, 142, 271},
    //                             {366, 285, 357, 485, 71, 470, 314, 110, 283, 193, 340, 342, 27, 257},
    //                             {294, 468, 404, 485, 336, 126, 486, 70, 27, 342, 457, 193, 42, 74},
    //                             {462, 126, 404, 202, 115, 71, 453, 118, 26, 193, 186, 138, 2, 349},
    //                             {362, 434, 486, 485, 32, 258, 440, 439, 70, 257, 146, 382, 142, 487},
    //                             {314, 146, 406, 486, 362, 485, 222, 257, 27, 118, 271, 87, 228, 274},
    //                             {434, 362, 406, 486, 485, 336, 202, 26, 70, 27, 439, 257, 64, 271},
    //                             {308, 404, 468, 336, 211, 406, 486, 162, 26, 118, 439, 27, 342, 257},
    //                             {469, 485, 486, 362, 434, 336, 326, 90, 87, 26, 70, 118, 293, 42},
    //                             {449, 404, 468, 485, 119, 154, 486, 42, 26, 118, 439, 27, 342, 228},
    //                             {135, 71, 406, 404, 119, 343, 154, 257, 27, 118, 146, 265, 242, 246},
    //                             {200, 120, 336, 468, 485, 294, 486, 295, 293, 26, 70, 118, 439, 246},
    //                             {343, 119, 154, 485, 406, 486, 362, 274, 228, 26, 293, 342, 498, 453},
    //                             {362, 406, 486, 485, 434, 336, 468, 274, 70, 26, 118, 439, 342, 228},
    //                             {119, 343, 154, 485, 406, 486, 362, 110, 70, 293, 246, 64, 87, 257},
    //                             {434, 362, 406, 71, 399, 283, 366, 443, 193, 293, 26, 70, 27, 450},
    //                             {154, 119, 485, 486, 406, 362, 258, 88, 118, 293, 85, 369, 457, 258},
    //                             {309, 486, 485, 119, 285, 222, 115, 444, 439, 118, 26, 70, 27, 42},
    //                             {406, 362, 71, 326, 399, 384, 343, 130, 26, 70, 27, 118, 439, 42},
    //                             {285, 486, 222, 119, 404, 468, 115, 378, 42, 293, 26, 70, 27, 118},
    //                             {362, 154, 452, 468, 406, 294, 76, 457, 70, 27, 118, 342, 246, 302}};
    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;

    // int maskLength = 5;  // for cyber 400
    int maskLength = 8;  // for knap100
    // int maskLength = 10;  // for knap250
    // int maskLength = 14;  // for knap500
    if(number_of_parameters == 100){
        maskLength = 8;
    }
    else if(number_of_parameters == 100){
        maskLength = 10;
    }

    for( i = 0; i < 12; i++ )
    // for( i = 0; i < 0; i++ )
    {
        // for cyber 400
        /*
        maskLength += 5; 
        if (maskLength > 60) {
            maskLength = 60;
        }
        */

        /* Pick donor from randomly choosed cluster */
        // cluster_index = randomInt(array_of_number_of_clusters[0]);
        // donor_index = randomInt( population_cluster_sizes[cluster_index] );
        // donor = population[population_indices_of_cluster_members[cluster_index][donor_index]];

        donor_index = randomInt(smallest_population_size);

        int mask_index = randomInt(49);  // knap 250 & 100
        // int mask_index = randomInt(39);  // knap 500

        donor = population[donor_index];

        /* Cheating BB mixing */
        linkage_group_index = 0;
        for (int _i = 0; _i < maskLength; _i++) {

            // for cyber
            // int parameter_index = cheatingMask[cluster_index][linkage_group_index][_i];
            // for knap
            int parameter_index = knapCheatMask[mask_index][_i];
            result[parameter_index] = donor[parameter_index];
        }
        
        // mutateSolution(result, linkage_group_index, cluster_index);

        /* Check if the new intermediate solution is different from the previous state. */
        is_unchanged = TRUE;
        for( j = 0; j < maskLength; j++ ) {
            // for cyber
            // if( backup[cheatingMask[cluster_index][linkage_group_index][j]] != result[cheatingMask[cluster_index][linkage_group_index][j]] ) {
            
            // for knap
            if( backup[knapCheatMask[mask_index][j]] != result[knapCheatMask[mask_index][j]] ) {
                is_unchanged = FALSE;
                break;
            }
        }

        if( is_unchanged == FALSE ) {
            is_improved = FALSE;

            evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);

            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, CHEATING_OM, chromosome_index, from_which);

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_improved = TRUE;
            
            if ( is_improved ) {
                changed = TRUE;
                *from_which = 6;
                
                copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
        }
        // if ( is_improved ) {
        //     break;
        // }


    }

    free(order);

    /* Forced Improvement */
    if (  (!changed) || (t_NIS > (1+floor(log10(population_size))))  ) {
        total_FI_times++;
        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++) {
            donor_index = randomInt(elitist_archive_size);
            donor = elitist_archive[donor_index];

            /* Use LT to do FI */
            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_index);

            /* Use cheating mask to do FI */
            // int mask_index = randomInt(39);
            // for (int _i = 0; _i < maskLength; _i++) {
            //     int parameter_index = knapCheatMask[mask_index][_i];
            //     result[parameter_index] = donor[parameter_index];
            // }

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ ) {
                if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] ) {
                    is_unchanged = FALSE;
                    break;
                }
            }

            if( is_unchanged == FALSE ) {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, MIDDLE_FI, chromosome_index, from_which);

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    *from_which = MIDDLE_FI;

                    copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
            }
        }
        free(order);

        if(!changed) {
            donor_index = randomInt( elitist_archive_size );
            copyFromAToB(elitist_archive[donor_index], elitist_archive_objective_values[donor_index], 
                    elitist_archive_constraint_values[donor_index], 
                    result, obj, con);
        }
    }

    free( backup ); 
    free( obj_backup ); 
}




/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *Malloc( long size )
{
    void *result;

    result = (void *) malloc( size );
    if( !result )
    {
        printf("\n");
        printf("Error while allocating memory in Malloc( %ld ), aborting program.", size);
        printf("\n");

        exit( 0 );
    }

    return( result );
}
/**
 * Initializes the pseudo-random number generator.
 */
void initializeRandomNumberGenerator()
{
    struct timeval tv;
    struct tm *timep;

    while( random_seed_changing == 0 )
    {
        gettimeofday( &tv, NULL );
        timep = localtime (&tv.tv_sec);
        random_seed_changing = timep->tm_hour * 3600 * 1000 + timep->tm_min * 60 * 1000 + timep->tm_sec * 1000 + tv.tv_usec / 1000;
    }
    random_seed_changing += runTimes;
    random_seed = random_seed_changing;
    printf("random seed: %ld\n", random_seed);
}
/**
 * Returns a random double, distributed uniformly between 0 and 1.
 */
double randomRealUniform01( void )
{
    int64_t n26, n27;
    double  result;

    random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    n26                  = (int64_t)(random_seed_changing >> (48 - 26));
    random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    n27                  = (int64_t)(random_seed_changing >> (48 - 27));
    result               = (((int64_t)n26 << 27) + n27) / ((double) (1LLU << 53));

    return( result );
}
        
/**
 * Returns a random integer, distributed uniformly between 0 and maximum.
 */
int randomInt( int maximum )
{
    int result;
    result = (int) (((double) maximum)*randomRealUniform01());
    return( result );
}
/**
 * Computes the two-log of x.
 */
double math_log_two = log(2.0);
double log2( double x )
{
  return( log(x) / math_log_two );
}
int* createRandomOrdering(int size_of_the_set)
{
    int *order, a, b, c, i;

    order = (int *) Malloc( size_of_the_set*sizeof( int ) );
    for( i = 0; i < size_of_the_set; i++ )
        order[i] = i;
    for( i = 0; i < size_of_the_set; i++ )
    {
        a        = randomInt( size_of_the_set );
        b        = randomInt( size_of_the_set );
        c        = order[a];
        order[a] = order[b];
        order[b] = c;
    }

    return order;
}
/**
 * Computes the Euclidean distance between two points.
 */
double distanceEuclidean( double *x, double *y, int number_of_dimensions )
{
    int    i;
    double value, result;

    result = 0.0;
    for( i = 0; i < number_of_dimensions; i++ )
    {
        value   = y[i] - x[i];
        result += value*value;
    }
    result = sqrt( result );

    return( result );
}
/**
 * Sorts an array of doubles and returns the sort-order (small to large).
 */
int *mergeSort( double *array, int array_size )
{
    int i, *sorted, *tosort;

    sorted = (int *) Malloc( array_size * sizeof( int ) );
    tosort = (int *) Malloc( array_size * sizeof( int ) );
    for( i = 0; i < array_size; i++ )
    tosort[i] = i;

    if( array_size == 1 )
        sorted[0] = 0;
    else
        mergeSortWithinBounds( array, sorted, tosort, 0, array_size-1 );

    free( tosort );

    return( sorted );
}
/**
 * Subroutine of merge sort, sorts the part of the array between p and q.
 */
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q )
{
    int r;

    if( p < q )
    {
        r = (p + q) / 2;
        mergeSortWithinBounds( array, sorted, tosort, p, r );
        mergeSortWithinBounds( array, sorted, tosort, r+1, q );
        mergeSortMerge( array, sorted, tosort, p, r+1, q );
    }
}
/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q )
{
    int i, j, k, first;

    i = p;
    j = r;
    for( k = p; k <= q; k++ )
    {
        first = 0;
        if( j <= q )
        {
            if( i < r )
            {
                if( array[tosort[i]] < array[tosort[j]] )
                first = 1;
            }
        }
        else
            first = 1;

        if( first )
        {
            sorted[k] = tosort[i];
            i++;
        }
        else
        {
            sorted[k] = tosort[j];
            j++;
        }
    }

    for( k = p; k <= q; k++ )
        tosort[k] = sorted[k];
}
/*-=-=-=-=-=-=-=-=-=-=- Section Interpret Command Line -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Parses and checks the command line.
 */
void interpretCommandLine( int argc, char **argv )
{
    parseCommandLine( argc, argv );
  
    checkOptions();
}
/**
 * Parses the command line.
 * For options, see printUsage.
 */
void parseCommandLine( int argc, char **argv )
{
    int index;

    index = 1;

    parseOptions( argc, argv, &index );
  
    parseParameters( argc, argv, &index );
}
/**
 * Parses only the options from the command line.
 */
void parseOptions( int argc, char **argv, int *index )
{
    double dummy;

    print_verbose_overview        = 0;
    use_vtr                       = 0;
    use_print_progress_to_screen  = 0;

    use_pre_mutation              = 0;
    use_pre_adaptive_mutation     = 0;
    use_repair_mechanism          = 0;
    stop_population_when_front_is_covered = 0;

    for( ; (*index) < argc; (*index)++ )
    {
        if( argv[*index][0] == '-' )
        {
            /* If it is a negative number, the option part is over */
            if( sscanf( argv[*index], "%lf", &dummy ) && argv[*index][1] != '\0' )
                break;

            if( argv[*index][1] == '\0' )
                optionError( argv, *index );
            else if( argv[*index][2] != '\0' )
                optionError( argv, *index );
            else
            {
                switch( argv[*index][1] )
                {
                    case '?': printUsage(); break;
                    case 'P': printAllInstalledProblems(); break;
                    case 'v': print_verbose_overview        = 1; break;
                    case 'p': use_print_progress_to_screen  = 1; break;
                    case 'm': use_pre_mutation              = 1; break;
                    case 'M': use_pre_adaptive_mutation     = 1; break;
                    case 'r': use_repair_mechanism          = 1; break; 
                    case 'z': stop_population_when_front_is_covered = 1; break;
                    default : optionError( argv, *index );
                }
            }
        }
        else /* Argument is not an option, so option part is over */
            break;
    }
}
/**
 * Writes the names of all installed problems to the standard output.
 */
void printAllInstalledProblems( void )
{
    int i, n;
    
    n = numberOfInstalledProblems();
    printf("Installed optimization problems:\n");
    for( i = 0; i < n; i++ )
        printf("%3d: %s\n", i, installedProblemName( i ));

    exit( 0 );
}
/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError( char **argv, int index )
{
    printf("Illegal option: %s\n\n", argv[index]);
    printUsage();
}
/**
 * Parses only the EA parameters from the command line.
 */
void parseParameters( int argc, char **argv, int *index )
{
    int noError;

    if( (argc - *index) != 9 )
    {
        printf("Number of parameters is incorrect, require 9 parameters (you provided %d).\n\n", (argc - *index));
        printUsage();
    }

    noError = 1;
    noError = noError && sscanf( argv[*index+0], "%d", &problem_index );
    noError = noError && sscanf( argv[*index+1], "%d", &number_of_objectives );
    noError = noError && sscanf( argv[*index+2], "%d", &number_of_parameters );
    noError = noError && sscanf( argv[*index+3], "%d", &elitist_archive_size_target );
    noError = noError && sscanf( argv[*index+4], "%d", &maximum_number_of_evaluations );
    noError = noError && sscanf( argv[*index+5], "%d", &log_progress_interval );
    noError = noError && sscanf( argv[*index+6], "%d", &runTimes );
    noError = noError && sscanf( argv[*index+7], "%lf", &antiProb );
    exp_name = (char*)Malloc(200 * sizeof(char));
    noError = noError && sscanf( argv[*index+8], "%s", exp_name );

    if( !noError )
    {
        printf("Error parsing parameters.\n\n");
        printUsage();
    }
    
    // Set output dir
    output_dir = (char*)Malloc(200 * sizeof(char));
    switch(problem_index){
        case ZEROMAX_ONEMAX:
            sprintf( output_dir, "zmom%d_result/%s/", number_of_parameters, exp_name);
            break;
        case TRAP5:
            sprintf( output_dir, "trap%d_result/%s/", number_of_parameters, exp_name);
            break;
        case KNAPSACK:
            sprintf( output_dir, "knapsack%d_result/%s/", number_of_parameters, exp_name);
            break;
        case LOTZ:
            sprintf( output_dir, "lotz%d_result/%s/", number_of_parameters, exp_name);
            break;
        case MAXCUT:
            sprintf( output_dir, "maxcut%d_result/%s/", number_of_parameters, exp_name);
            break;
        default:
            printf("Output dir not set!\n");
            exit(1);
    }

}
/**
 * Prints usage information and exits the program.
 */
void printUsage( void )
{
    printf("Usage: MO-GOMEA [-?] [-P] [-s] [-w] [-v] [-r] [-g] pro dim eas eva log gen run_time antimutate_prob exp_name\n");
    printf(" -?: Prints out this usage information.\n");
    printf(" -P: Prints out a list of all installed optimization problems.\n");
    printf(" -p: Prints optimization progress to screen.\n");
    printf(" -v: Enables verbose mode. Prints the settings before starting the run.\n");
    printf(" -r: Enables use of a repair mechanism if the problem is constrained.\n");
    printf(" -m: Enables use of the weak mutation operator.\n");
    printf(" -M: Enables use of the strong mutation operator.\n");
    printf(" -z: Enable checking if smaller (inefficient) populations should be stopped.\n");
    printf("\n");
    printf("  pro: Index of optimization problem to be solved.\n");
    printf("  num: Number of objectives to be optimized.\n");
    printf("  dim: Number of parameters.\n");
    printf("  eas: Elitist archive size target.\n");
    printf("  eva: Maximum number of evaluations allowed.\n");
    printf("  log: Interval (in terms of number of evaluations) at which the elitist archive is recorded for logging purposes.\n");
    exit( 0 );
}
/**
 * Checks whether the selected options are feasible.
 */
void checkOptions( void )
{
    if( elitist_archive_size_target < 1 )
    {
        printf("\n");
        printf("Error: elitist archive size target < 1 (read: %d).", elitist_archive_size_target);
        printf("\n\n");

        exit( 0 );
    }
    if( maximum_number_of_evaluations < 1 )
    {
        printf("\n");
        printf("Error: maximum number of evaluations < 1 (read: %d). Require maximum number of evaluations >= 1.", maximum_number_of_evaluations);
        printf("\n\n");

        exit( 0 );
    }
    if( installedProblemName( problem_index ) == NULL )
    {
        printf("\n");
        printf("Error: unknown index for problem (read index %d).", problem_index );
        printf("\n\n");

        exit( 0 );
    }
}
/**
 * Prints the settings as read from the command line.
 */
void printVerboseOverview( void )
{
    printf("###################################################\n");
    printf("#\n");
    printf("# Problem                 = %s\n", installedProblemName( problem_index ));
    printf("# Number of objectives    = %d\n", number_of_objectives);
    printf("# Number of parameters    = %d\n", number_of_parameters);
    printf("# Elitist ar. size target = %d\n", elitist_archive_size_target);
    printf("# Maximum numb. of eval.  = %d\n", maximum_number_of_evaluations);
    printf("# Random seed             = %ld\n", random_seed);
    printf("#\n");
    printf("###################################################\n");
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void evaluateIndividual(char *solution, double *obj, double *con, int objective_index_of_extreme_cluster)
{
    number_of_evaluations++;
    if(population_id != -1)
        array_of_number_of_evaluations_per_population[population_id] += 1;

    switch(problem_index)
    {
        case ZEROMAX_ONEMAX: onemaxProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case TRAP5: trap5ProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case LOTZ: lotzProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case KNAPSACK: knapsackProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case MAXCUT: maxcutProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        default:
            printf("Cannot evaluate this problem!\n");
            exit(1);
    }

    logElitistArchiveAtSpecificPoints();  
}
/**
 * Returns the name of an installed problem.
 */
char *installedProblemName( int index )
{
    switch( index )
    {
        case  ZEROMAX_ONEMAX:   return( (char *) "Zeromax - Onemax" );
        case  TRAP5:            return( (char *) "Deceptive Trap 5 - Inverse Trap 5 - Tight Encoding" );
        case  KNAPSACK:         return( (char *) "Knapsack - 2 Objectives");
        case  LOTZ:             return( (char *) "Leading One Trailing Zero (LOTZ)");
        case  MAXCUT:           return( (char *) "Maxcut - 2 Objectives");
    }
    return( NULL );
}
/**
 * Returns the number of problems installed.
 */
int numberOfInstalledProblems( void )
{
    static int result = -1;
  
    if( result == -1 )
    {
        result = 0;
        while( installedProblemName( result ) != NULL )
            result++;
    }
  
    return( result );
}

void onemaxLoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void trap5LoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void lotzLoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void onemaxProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i, number_of_1s, number_of_0s;
    
    *con_value = 0.0;
    number_of_0s = 0;
    number_of_1s = 0;
    
    for(i = 0; i < number_of_parameters; i++)
    {
        if(solution[i] == 0)
            number_of_0s++;
        else if(solution[i] == 1)
            number_of_1s++;
    }

    obj_values[0] = number_of_0s;
    obj_values[1] = number_of_1s;
}

double deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, int k, char is_one )
{
    int    i, j, m, u;
    double result;

    if( number_of_parameters % k != 0 )
    {
        printf("Error in evaluating deceptive trap k: Number of parameters is not a multiple of k.\n");
        exit( 0 );
    }

    m      = number_of_parameters / k;
    result = 0.0;
    for( i = 0; i < m; i++ )
    {
        u = 0;
        for( j = 0; j < k; j++ )
        u += (parameters[i*k+j] == is_one) ? 1 : 0;

        if( u == k )
            result += k;
        else
            result += (k-1-u);
    }

    return result;
}

void trap5ProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    *con_value      = 0.0;
    obj_values[0]   = deceptiveTrapKTightEncodingFunctionProblemEvaluation( solution, 5, TRUE );
    obj_values[1]   = deceptiveTrapKTightEncodingFunctionProblemEvaluation( solution, 5, FALSE );
}

void lotzProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i;
    double result;

    *con_value = 0.0;
    result = 0.0;
    for(i = 0; i < number_of_parameters; i++)
    {
        if(solution[i] == 0)
            break;
        result += 1;
    }
    obj_values[0] = result; // Leading Ones

    result = 0.0;
    for(i = number_of_parameters - 1; i >= 0; i--)
    {
        if(solution[i] == 1)
            break;
        result += 1;
    }
    obj_values[1] = result; // Trailing Zeros
}

void knapsackLoadProblemData()
{
    int int_number, i, k;
    FILE *file;
    char string[1000];
    double double_number, ratio, *ratios;

    sprintf(string, "./knapsack/knapsack.%d.%d.txt", number_of_parameters, number_of_objectives);
    file = NULL;
    file = fopen(string, "r");
    if(file == NULL)
    {
        printf("Cannot open file %s!\n", string);
        exit(1);
    }

    fscanf(file, "%d", &int_number);
    fscanf(file, "%d", &int_number);

    capacities      = (double*)Malloc(number_of_objectives*sizeof(double));
    weights         = (double**)Malloc(number_of_objectives*sizeof(double*));
    profits         = (double**)Malloc(number_of_objectives*sizeof(double*));
    for(k = 0; k < number_of_objectives; k++)
    {
        weights[k]  = (double*)Malloc(number_of_parameters*sizeof(double));
        profits[k]  = (double*)Malloc(number_of_parameters*sizeof(double));
    }
    for(k = 0; k < number_of_objectives; k++)
    {
        fscanf(file, "%lf", &double_number);
        capacities[k] = double_number;

        for(i = 0; i < number_of_parameters; i++)
        {
            fscanf(file, "%d", &int_number);
            
            fscanf(file, "%d", &int_number);
            weights[k][i] = int_number;
            fscanf(file, "%d", &int_number);
            profits[k][i] = int_number;
        }
    }
    fclose(file);

    ratio_profit_weight = (double*)Malloc(number_of_parameters*sizeof(double));
    for(i = 0; i < number_of_parameters; i++)
    {
        ratio_profit_weight[i] = profits[0][i] / weights[0][i];
        for(k = 1; k < number_of_objectives; k++)
        {
            ratio = profits[k][i] / weights[k][i];
            if(ratio > ratio_profit_weight[i])
                ratio_profit_weight[i] = ratio;
        }
    }

    item_indices_least_profit_order = mergeSort(ratio_profit_weight, number_of_parameters);
    item_indices_least_profit_order_according_to_objective = (int**)Malloc(number_of_objectives*sizeof(int*));
    ratios = (double*)Malloc(number_of_parameters*sizeof(double));
    for(k = 0; k < number_of_objectives; k++)
    {
        for(i = 0; i < number_of_parameters; i++)
            ratios[i] = profits[k][i] / weights[k][i];
        item_indices_least_profit_order_according_to_objective[k] = mergeSort(ratios, number_of_parameters);
    }

    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    free(ratios);
}

void ezilaitiniKnapsackProblemData()
{
    int k;
    for(k = 0; k < number_of_objectives; k++)
    {
        free(weights[k]);
        free(profits[k]);
    }
    free(weights);
    free(profits);
    free(capacities);
    free(ratio_profit_weight);
    free(item_indices_least_profit_order);

    for(k = 0; k < number_of_objectives; k++)
    {
        free(item_indices_least_profit_order_according_to_objective[k]);
    }
    free(item_indices_least_profit_order_according_to_objective);
}

void knapsackSolutionRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index_of_extreme_cluster)
{
    if(objective_index_of_extreme_cluster == -1)
        knapsackSolutionMultiObjectiveRepair(solution, solution_profits, solution_weights, solution_constraint);
    else
        knapsackSolutionSingleObjectiveRepair(solution, solution_profits, solution_weights, solution_constraint, objective_index_of_extreme_cluster);
}

void knapsackSolutionSingleObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index)
{
    int i, j, k;
    char isFeasible;

    for(j = 0; j < number_of_parameters; j++)
    {
        i = item_indices_least_profit_order_according_to_objective[objective_index][j];
        if(solution[i] == 0)
            continue;

        solution[i] = 0;
        isFeasible = TRUE;
        for(k = 0; k < number_of_objectives; k++)
        {
            solution_profits[k] = solution_profits[k] - profits[k][i];
            solution_weights[k] = solution_weights[k] - weights[k][i];
            if(solution_weights[k] > capacities[k])
                isFeasible = FALSE;
        }
        if(isFeasible == TRUE)
            break;
    }

    *solution_constraint = 0.0;
    for(k = 0; k < number_of_objectives; k++)
        if(solution_weights[k] > capacities[k])
            (*solution_constraint) = (*solution_constraint) + (solution_weights[k] - capacities[k]);    
}

void knapsackSolutionMultiObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint)
{
    int i, j, k;
    char isFeasible;

    for(j = 0; j < number_of_parameters; j++)
    {
        i = item_indices_least_profit_order[j];
        if(solution[i] == 0)
            continue;

        solution[i] = 0;
        isFeasible = TRUE;
        for(k = 0; k < number_of_objectives; k++)
        {
            solution_profits[k] = solution_profits[k] - profits[k][i];
            solution_weights[k] = solution_weights[k] - weights[k][i];
            if(solution_weights[k] > capacities[k])
                isFeasible = FALSE;
        }
        if(isFeasible == TRUE)
            break;
    }

    *solution_constraint = 0.0;
    for(k = 0; k < number_of_objectives; k++)
        if(solution_weights[k] > capacities[k])
            (*solution_constraint) = (*solution_constraint) + (solution_weights[k] - capacities[k]);
}

void knapsackProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i, k;
    double *solution_profits, *solution_weights;

    solution_weights = (double*)Malloc(number_of_objectives*sizeof(double));
    solution_profits = (double*)Malloc(number_of_objectives*sizeof(double));
    *con_value = 0.0;
    for(k = 0; k < number_of_objectives; k++)
    {
        solution_profits[k] = 0.0;
        solution_weights[k] = 0.0;
        for(i = 0; i < number_of_parameters; i++)
        {
            solution_profits[k] += ((int)solution[i])*profits[k][i];
            solution_weights[k] += ((int)solution[i])*weights[k][i];
        }
        if(solution_weights[k] > capacities[k])
            (*con_value) = (*con_value) + (solution_weights[k] - capacities[k]);
    }

    if(use_repair_mechanism)
    {
        if( (*con_value) > 0)
            knapsackSolutionRepair(solution, solution_profits, solution_weights, con_value, objective_index_of_extreme_cluster);
    }

    for(k = 0; k < number_of_objectives; k++)
        obj_values[k] = solution_profits[k];

    free(solution_weights);
    free(solution_profits);
}

void maxcutLoadProblemData()
{
    int i, k;
    char string[1000];
    maxcut_edges = (int ***) Malloc(number_of_objectives * sizeof(int **));
    number_of_maxcut_edges = (int *) Malloc(number_of_objectives * sizeof(int ));
    maxcut_edges_weights = (double **) Malloc(number_of_objectives * sizeof(double *));

    for (i = 0; i < number_of_objectives; i++)
    {
        sprintf(string, "maxcut/maxcut_instance_%d_%d.txt", number_of_parameters, i);
        maxcutReadInstanceFromFile(string, i);
    }

    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;
}

void ezilaitiniMaxcutProblemData()
{
    int i,j;
    for(i=0;i<number_of_objectives;i++)
    {
        for(j=0;j<number_of_maxcut_edges[i];j++)
            free(maxcut_edges[i][j]);
        free(maxcut_edges[i]);
        free(maxcut_edges_weights[i]);
    }
    free(maxcut_edges);
    free(maxcut_edges_weights);
    free(number_of_maxcut_edges); 
}

void maxcutReadInstanceFromFile(char *filename, int objective_index)
{
    char  c, string[1000], substring[1000];
    int   i, j, k, q, number_of_vertices, number_of_edges;
    FILE *file;

    //file = fopen( "maxcut_instance.txt", "r" );
    file = fopen( filename, "r" );
    if( file == NULL )
    {
        printf("Error in opening file \"maxcut_instance.txt\"");
        exit( 0 );
    }

    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
    }
    string[k] = '\0';

    q = 0;
    j = 0;
    while( (string[j] != ' ') && (j < k) )
    {
        substring[q] = string[j];
        q++;
        j++;
    }
    substring[q] = '\0';
    j++;

    number_of_vertices = atoi( substring );
    if( number_of_vertices != number_of_parameters )
    {
        printf("Error during reading of maxcut instance:\n");
        printf("  Read number of vertices: %d\n", number_of_vertices);
        printf("  Doesn't match number of parameters on command line: %d\n", number_of_parameters);
        exit( 1 );
    }

    q = 0;
    while( (string[j] != ' ') && (j < k) )
    {
        substring[q] = string[j];
        q++;
        j++;
    }
    substring[q] = '\0';
    j++;

    number_of_edges = atoi( substring );
    number_of_maxcut_edges[objective_index] = number_of_edges;
    maxcut_edges[objective_index] = (int **) Malloc( number_of_edges*sizeof( int * ) );
    for( i = 0; i < number_of_edges; i++ )
        maxcut_edges[objective_index][i] = (int *) Malloc( 2*sizeof( int ) );
    maxcut_edges_weights[objective_index] = (double *) Malloc( number_of_edges*sizeof( double ) );

    i = 0;
    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
    }
    string[k] = '\0';
    while( k > 0 )
    {
        q = 0;
        j = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges[objective_index][i][0] = atoi( substring )-1;

        q = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges[objective_index][i][1] = atoi( substring )-1;

        q = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges_weights[objective_index][i] = atof( substring );
        i++;

        c = fgetc( file );
        k = 0;
        while( c != '\n' && c != EOF )
        {
            string[k] = (char) c;
            c      = fgetc( file );
            k++;
        }
        string[k] = '\0';
    }

    fclose( file );
}

void maxcutProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster )
{
    int    i, k;
    double result;

    *con_value = 0;

    for(k = 0; k < number_of_objectives; k++)
    {
        result = 0.0;
        for( i = 0; i < number_of_maxcut_edges[k]; i++ )
        {
            if( solution[maxcut_edges[k][i][0]] != solution[maxcut_edges[k][i][1]] )
                result += maxcut_edges_weights[k][i];
        }

        obj_values[k] = result;
    }
}

double **getDefaultFrontOnemaxZeromax( int *default_front_size )
{
    int  i;
    static double **result = NULL;
    *default_front_size = ( number_of_parameters + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )
        {
            result[i][0] = i;                                         // Zeromax
            result[i][1] = number_of_parameters - result[i][0];       // Onemax
        }
    }
    return( result );
}

double **getDefaultFrontTrap5InverseTrap5( int *default_front_size )
{
    int  i, number_of_blocks;
    static double **result = NULL;

    number_of_blocks = number_of_parameters / 5;
    *default_front_size = ( number_of_blocks + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )                    // i = number of all-1 blocks
        {
            result[i][0] = ( 5 * i ) + ( 4 * (number_of_blocks - i) ) ;   // Trap-5
            result[i][1] = ( 5 * (number_of_blocks - i)) + ( 4 * i );     // Inverse Trap-5
        }
    }
    return( result );
}

double **getDefaultFrontLeadingOneTrailingZero( int *default_front_size )
{
    int  i;
    static double **result = NULL;

    *default_front_size = ( number_of_parameters + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )
        {
            result[i][0] = i;                                         // Leading One
            result[i][1] = number_of_parameters - result[i][0];       // Trailing Zero
        }
    }

    return( result );
}
/**
 * Returns whether the D_{Pf->S} metric can be computed.
 */
short haveDPFSMetric( void )
{
    int default_front_size;

    getDefaultFront( &default_front_size );
    if( default_front_size > 0 )
        return( 1 );

    return( 0 );
}
/**
 * Returns the default front(NULL if there is none).
 * The number of solutions in the default
 * front is returned in the pointer variable.
 */
double **getDefaultFront( int *default_front_size )
{
    switch( problem_index )
    {
        case ZEROMAX_ONEMAX: return( getDefaultFrontOnemaxZeromax( default_front_size ) );
        case TRAP5: return( getDefaultFrontTrap5InverseTrap5( default_front_size ) );
        case LOTZ: return( getDefaultFrontLeadingOneTrailingZero( default_front_size ) );
    }

    *default_front_size = 0;
    return( NULL );
}

double computeDPFSMetric( double **default_front, int default_front_size, double **approximation_front, int approximation_front_size )
{
    int    i, j;
    double result, distance, smallest_distance;

    if( approximation_front_size == 0 )
        return( 1e+308 );

    result = 0.0;
    for( i = 0; i < default_front_size; i++ )
    {
        smallest_distance = 1e+308;
        for( j = 0; j < approximation_front_size; j++ )
        {
            distance = distanceEuclidean( default_front[i], approximation_front[j], number_of_objectives );
            if( distance < smallest_distance )
                smallest_distance = distance;
      
        }
        result += smallest_distance;
    }
    result /= (double) default_front_size;

    return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Tracking Progress =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */
void writeGenerationalStatistics( void )
{
    int     i;
    char    string[1000];
    FILE   *file;

    file = NULL;
    if(( number_of_generations == 0 && population_id == 0) ||
        (number_of_generations == 0 && population_id == -1))
    {
        file = fopen( "statistics.dat", "w" );

        sprintf( string, "# Generation  Population  Evaluations   [ Cluster_Index ]\n");
        fputs( string, file );
    }
    else
        file = fopen( "statistics.dat", "a" );

    sprintf( string, "  %10d %10d %11d     [ ", number_of_generations, population_size, number_of_evaluations );
    fputs( string, file );

    for( i = 0; i < number_of_mixing_components; i++ )
    {
        sprintf( string, "%4d", i );
        fputs( string, file );
        if( i < number_of_mixing_components-1 )
        {
            sprintf( string, " " );
            fputs( string, file );
        }
    }
    sprintf( string, " ]\n" );
    fputs( string, file );

    fclose( file );

    freeAuxiliaryPopulations();
}

void writeCurrentElitistArchive( char final )
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;

    /* Elitist archive */
    if( final )
        sprintf( string, "%sresult%d/elitist_archive_generation_final.dat" );
    else
        sprintf( string, "%sresult%d/elitist_archive_at_evaluation_%d.dat", output_dir, runTimes, number_of_evaluations );

    file = fopen( string, "w" );

    for( i = 0; i < elitist_archive_size; i++ )
    {
        for( j = 0; j < number_of_objectives; j++ )
        {
            sprintf( string, "%13e ", elitist_archive_objective_values[i][j] );
            fputs( string, file );
        }

        sprintf( string, "   %f     ", elitist_archive_constraint_values[i]);
        fputs( string, file );

        for( j = 0; j < number_of_parameters; j++ )
        {
            sprintf( string, "%d ", elitist_archive[i][j] );
            fputs( string, file );
        }
        sprintf( string, "\n" );
        fputs( string, file );
    }
    fclose( file );
}

void writeCurrentAngle(double angle, bool is_FI)
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;
    if(is_FI)
        sprintf( string, "%sresult%d/angles_FI.dat", output_dir, runTimes, number_of_evaluations );
    else
        sprintf( string, "%sresult%d/angles_GOM.dat", output_dir, runTimes, number_of_evaluations );

    file = fopen( string, "a" );
    sprintf( string, "%13e ", angle );
    fputs( string, file );
    fclose( file );
}


void writeCurrentLevel(double level, bool is_FI)
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;
    if(is_FI)
        sprintf( string, "%sresult%d/level.dat", output_dir, runTimes, number_of_evaluations );
        
    file = fopen( string, "a" );
    sprintf( string, "%13e ", level );
    fputs( string, file );
    fclose( file );
}



void writeCurrentMask(int cluster_index, int linkage_group_index, char *result, char *backup)
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;
    sprintf( string, "%sresult%d/masks_FI.dat", output_dir, runTimes, number_of_evaluations );
    file = fopen( string, "a" );
    for( j = 0; j < number_of_parameters; j++){
        i = 0;
        int check = -1;
        for(i = 0; i < lt_number_of_indices[cluster_index][linkage_group_index]; i++){
            if(j == lt[cluster_index][linkage_group_index][i]){
                check = i;
                break;
            }
        }
        if( check != -1 ) {
            sprintf( string, "%d", result[lt[cluster_index][linkage_group_index][check]] );
            fputs( string, file );
            i++;
        }
        else {
            sprintf( string, "*" );
            fputs( string, file );
        }
    }
    sprintf( string, "\n");
    fputs( string, file );
    fclose( file );
}



void writeCurrentNotInClusterRatio(int not_in_cluster_cnt, int total_cnt)
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;
    sprintf( string, "%sresult%d/notInClusterRatio.dat", output_dir, runTimes, number_of_evaluations );
    file = fopen( string, "a" );
    sprintf( string, "%ld/%ld\n", not_in_cluster_cnt, total_cnt);
    fputs( string, file );
    fclose( file );
}



void writeCurrentChangeBits(int len, bool is_FI, int mask)
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;
    if(is_FI)
        sprintf( string, "%sresult%d/bits_FI.dat", output_dir, runTimes, number_of_evaluations );
    else
        sprintf( string, "%sresult%d/bits_GOM.dat", output_dir, runTimes, number_of_evaluations );

    file = fopen( string, "a" );
    sprintf( string, "%d ", len );
    fputs( string, file );
    fclose( file );
}


void writeCurrentElitistArchiveFrom( char final )
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;

    /* Elitist archive */
    if( final )
        sprintf( string, "elitist_archive_from_generation_final.dat" );
    else
        sprintf( string, "%sresult%d/elitist_archive_from_at_evaluation_%d.dat", output_dir, runTimes, number_of_evaluations );
    file = fopen( string, "w" );
    int cnt[8] ={0,0,0,0,0,0,0,0};
    
    for( i = 0; i < elitist_archive_size; i++ ) {
        cnt[elitist_archive_from[i]]++;
    }
    for(i = 0; i < 8; i++) {
        sprintf( string, "%d: %d\n",i, cnt[i]);
        fputs( string, file );
    }
    fclose( file );
}

void writeCurrentHistory()
{
    int   i, j, k, index;
    char  string[1000];
    FILE *file;
    sprintf( string, "%sresult%d/history.dat", output_dir, runTimes, number_of_evaluations );

    file = fopen( string, "w" );
    for(i = 0; i < elitist_archive_size; i ++){
        int idx = elitist_archive_chromosome_index[i];
        for(j = 0; j <= elitist_archive_evolution_due[i]; j++){
            sprintf( string, "%d",history[idx][j]);
            fputs( string, file );
        }
        sprintf( string, "\n");
        fputs( string, file );
    }
    fclose( file );

    sprintf( string, "%sresult%d/history_all.dat", output_dir, runTimes, number_of_evaluations );
    file = fopen( string, "w" );
    for(i = 0; i < history.size(); i ++){
        for(j = 0; j < history[i].size(); j++){
            sprintf( string, "%d",history[i][j]);
            fputs( string, file );
        }
        sprintf( string, "\n");
        fputs( string, file );
    }
    
    fclose( file );
}

void logElitistArchiveAtSpecificPoints()
{
    // if(number_of_evaluations%log_progress_interval == 0)
    if(number_of_evaluations%50000 == 0){
        writeCurrentElitistArchive( FALSE );
        writeCurrentElitistArchiveFrom( FALSE );
        elitist_num_history.push_back(elitist_archive_size);
    }
}
/**
 * Returns TRUE if termination should be enforced, FALSE otherwise.
 */
char checkTerminationCondition()
{
    if( maximum_number_of_evaluations >= 0 )
    {
        if( checkNumberOfEvaluationsTerminationCondition() )
            return( TRUE );
    }

    if( use_vtr )
    {
        if( checkVTRTerminationCondition() )
            return( TRUE );
    }

    return( FALSE );
}
/**
 * Returns TRUE if the maximum number of evaluations
 * has been reached, FALSE otherwise.
 */
char checkNumberOfEvaluationsTerminationCondition()
{
  if( number_of_evaluations >= maximum_number_of_evaluations )
    return( TRUE );

  return( FALSE );
}
/**
 * Returns 1 if the value-to-reach has been reached
 * for the multi-objective case. This means that
 * the D_Pf->S metric has reached the value-to-reach.
 * If no D_Pf->S can be computed, 0 is returned.
 */
char checkVTRTerminationCondition( void )
{
  int      default_front_size;
  double **default_front, metric_elitist_archive;

  if( haveDPFSMetric() )
  {
    default_front          = getDefaultFront( &default_front_size );
    metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive_objective_values, elitist_archive_size );

    if( metric_elitist_archive <= vtr )
    {
      return( 1 );
    }
  }

  return( 0 );
}

void logNumberOfEvaluationsAtVTR()
{
    int      default_front_size;
    double **default_front, metric_elitist_archive;
    FILE *file;
    char string[1000];

    if(use_vtr == FALSE)
        return;

    if( haveDPFSMetric() )
    {
        default_front          = getDefaultFront( &default_front_size );
        metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive_objective_values, elitist_archive_size );

        sprintf(string, "number_of_evaluations_when_all_points_found_%d.dat", output_dir);
        file = fopen(string, "a");
        if( metric_elitist_archive <= vtr )
        {
            fprintf(file, "%d\n", number_of_evaluations);
        }
        else
        {
            fprintf(file, "Cannot find all points within current budget!\n");    
        }
        fclose(file);  
    }
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=  Elitist Archive -==-=-=-=-=-=-=-=-=-=-=-=*/
char isDominatedByElitistArchive( double *obj, double con, char *is_new_nondominated_point, int *position_of_existed_member )
{
    int j;

    *is_new_nondominated_point = TRUE;
    *position_of_existed_member = -1;
    for( j = 0; j < elitist_archive_size; j++ )
    {
        if( constraintParetoDominates( elitist_archive_objective_values[j], elitist_archive_constraint_values[j], obj, con ) )
        {
            *is_new_nondominated_point = FALSE;
            return( TRUE );
        }
        else
        {
            if( !constraintParetoDominates( obj, con, elitist_archive_objective_values[j], elitist_archive_constraint_values[j] ) )
            {
              if( sameObjectiveBox( elitist_archive_objective_values[j], obj ) )
              {
                *is_new_nondominated_point = FALSE;
                *position_of_existed_member = j;
                return( FALSE );
              }
            }
        }
    }
    return( FALSE );
}
/**
 * Returns 1 if two solutions share the same objective box, 0 otherwise.
 */
short sameObjectiveBox( double *objective_values_a, double *objective_values_b )
{
    int i;

    if( !objective_discretization_in_effect )
    {
        /* If the solutions are identical, they are still in the (infinitely small) same objective box. */
        for( i = 0; i < number_of_objectives; i++ )
        {
            if( objective_values_a[i] != objective_values_b[i] )
                return( 0 );
        }

        return( 1 );
    }


    for( i = 0; i < number_of_objectives; i++ )
    {
        if( ((int) (objective_values_a[i] / objective_discretization[i])) != ((int) (objective_values_b[i] / objective_discretization[i])) )
            return( 0 );
    }

    return( 1 );
}

int hammingDistanceInParameterSpace(char *solution_1, char *solution_2)
{
	int i, distance;
	distance=0;
	for (i=0; i < number_of_parameters; i++)
	{
		if( solution_1[i] != solution_2[i])
			distance++;
	}

	return distance;
}

int hammingDistanceToNearestNeighborInParameterSpace(char *solution, int replacement_position)
{
	int i, distance_to_nearest_neighbor, distance;
	distance_to_nearest_neighbor = -1;
	for (i = 0; i < elitist_archive_size; i++)
	{
		if (i != replacement_position)
		{
			distance = hammingDistanceInParameterSpace(solution, elitist_archive[i]);
			if (distance < distance_to_nearest_neighbor || distance_to_nearest_neighbor < 0)
				distance_to_nearest_neighbor = distance;
		}
	}

	return distance_to_nearest_neighbor;
}
/**
 * Updates the elitist archive by offering a new solution
 * to possibly be added to the archive. If there are no
 * solutions in the archive yet, the solution is added.
 * Solution A is always dominated by solution B that is
 * in the same domination-box if B dominates A or A and
 * B do not dominate each other. If the solution is not
 * dominated, it is added to the archive and all solutions
 * dominated by the new solution, are purged from the archive.
 */
void updateElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, int solution_from, int chromosome_index)
{
    short is_dominated_itself;
    int   i, *indices_dominated, number_of_solutions_dominated;

    if( elitist_archive_size == 0 )
        addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_from, chromosome_index);
    else
    {
        indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
        number_of_solutions_dominated = 0;
        is_dominated_itself           = 0;
        for( i = 0; i < elitist_archive_size; i++ )
        {
            if( constraintParetoDominates( elitist_archive_objective_values[i], elitist_archive_constraint_values[i], solution_objective_values, solution_constraint_value ) )
                is_dominated_itself = 1;
            else
            {
                if( !constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    if( sameObjectiveBox( elitist_archive_objective_values[i], solution_objective_values ) )
                        is_dominated_itself = 1;
                }
            }

            if( is_dominated_itself )
                break;
        }

        if( !is_dominated_itself )
        {
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    indices_dominated[number_of_solutions_dominated] = i;
                    number_of_solutions_dominated++;
                }
            }

            if( number_of_solutions_dominated > 0 )
                removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );

            addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_from, chromosome_index);
        }

        free( indices_dominated );
    }
}
void updateElitistArchiveWithReplacementOfExistedMember( char *solution, double *solution_objective_values, double solution_constraint_value, char *is_new_nondominated_point, char *is_dominated_by_archive, int solution_from, int chromosome_index,  int *from_which)
{
    short is_existed, index_of_existed_member;
    int   i, *indices_dominated, number_of_solutions_dominated;
    int distance_old, distance_new;

    *is_new_nondominated_point  = TRUE;
    *is_dominated_by_archive    = FALSE;

    if( elitist_archive_size == 0 )
        addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_from, chromosome_index);
    else
    {
        indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
        number_of_solutions_dominated = 0;
        is_existed					  = 0;
        for( i = 0; i < elitist_archive_size; i++ )
        {
            if( constraintParetoDominates( elitist_archive_objective_values[i], elitist_archive_constraint_values[i], solution_objective_values, solution_constraint_value ) )
            {
                *is_dominated_by_archive    = TRUE;
                *is_new_nondominated_point  = FALSE;
            }
            else
            {
                if( !constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    if( sameObjectiveBox( elitist_archive_objective_values[i], solution_objective_values ) )
                    {
                        is_existed                  = 1;
                        index_of_existed_member     = i;
                        *is_new_nondominated_point  = FALSE;
                    }
                }
            }

            if( (*is_new_nondominated_point) == FALSE )
                break;
        }

        if( (*is_new_nondominated_point) == TRUE )
        {
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    indices_dominated[number_of_solutions_dominated] = i;
                    number_of_solutions_dominated++;
                }
            }

            if( number_of_solutions_dominated > 0 )
                removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );

            *from_which = solution_from;
            addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_from, chromosome_index);
            elitist_archive_front_changed = TRUE;
        }

        if( is_existed )
        {
            /* TRY KEEP SAME OBJ */
            
            // int dist = hammingDistanceInParameterSpace(solution, elitist_archive[index_of_existed_member]);
            // if(dist != 0){
            //     *from_which = solution_from;
            //     addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_from, chromosome_index);
            // }
            
            /* ORIGINAL SAME OBJ HANDLING */
            
            distance_old = hammingDistanceToNearestNeighborInParameterSpace(elitist_archive[index_of_existed_member], index_of_existed_member);
            distance_new = hammingDistanceToNearestNeighborInParameterSpace(solution, index_of_existed_member);
            if (distance_new > distance_old)
            {
                *from_which = solution_from;
                for(i = 0; i < number_of_parameters; i++)
                    elitist_archive[index_of_existed_member][i] = solution[i];
                for(i=0; i < number_of_objectives; i++)
                    elitist_archive_objective_values[index_of_existed_member][i] = solution_objective_values[i];
                elitist_archive_constraint_values[index_of_existed_member] = solution_constraint_value;
                elitist_archive_from[index_of_existed_member] = solution_from;
                elitist_archive_chromosome_index[index_of_existed_member] = chromosome_index;
                elitist_archive_evolution_due[index_of_existed_member] = gen_now;
            }
        }
        free( indices_dominated );
    }
}

/**
 * Removes a set of solutions (identified by their archive-indices)
 * from the elitist archive.
 */
void removeFromElitistArchive( int *indices, int number_of_indices )
{
    int      i, j, elitist_archive_size_new;
    char **elitist_archive_new;
    double **elitist_archive_objective_values_new;
	double *elitist_archive_constraint_values_new;
    int *elitist_archive_from_new;
    int *elitist_archive_chromosome_index_new;
    int *elitist_archive_evolution_due_new;

    elitist_archive_new                   = (char**) Malloc( elitist_archive_capacity*sizeof( char * ) );
    elitist_archive_objective_values_new  = (double **) Malloc( elitist_archive_capacity*sizeof( double * ) );
    elitist_archive_constraint_values_new = (double *) Malloc( elitist_archive_capacity*sizeof( double ) );
    elitist_archive_from_new = (int *) Malloc( elitist_archive_capacity*sizeof( int ) );
    elitist_archive_chromosome_index_new = (int *) Malloc( elitist_archive_capacity*sizeof( int ) );
    elitist_archive_evolution_due_new = (int *) Malloc( elitist_archive_capacity*sizeof(int));

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        elitist_archive_new[i]                  = (char *) Malloc( number_of_parameters*sizeof( char ) );
        elitist_archive_objective_values_new[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }

    elitist_archive_size_new = 0;
    for( i = 0; i < elitist_archive_size; i++ )
    {
        if( !isInListOfIndices( i, indices, number_of_indices ) )
        {
            for( j = 0; j < number_of_parameters; j++ )
                elitist_archive_new[elitist_archive_size_new][j] = elitist_archive[i][j];
            for( j = 0; j < number_of_objectives; j++ )
                elitist_archive_objective_values_new[elitist_archive_size_new][j] = elitist_archive_objective_values[i][j];
            elitist_archive_constraint_values_new[elitist_archive_size_new] = elitist_archive_constraint_values[i];
            elitist_archive_from_new[elitist_archive_size_new] = elitist_archive_from[i];
            elitist_archive_chromosome_index_new[elitist_archive_size_new] = elitist_archive_chromosome_index[i];
            elitist_archive_evolution_due_new[elitist_archive_size_new] = elitist_archive_evolution_due[i];
            elitist_archive_size_new++;
        }
    }

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        free( elitist_archive[i] );
        free( elitist_archive_objective_values[i] );
    }
    free( elitist_archive );
    free( elitist_archive_objective_values );
    free( elitist_archive_constraint_values );
    free( elitist_archive_from );
    free( elitist_archive_chromosome_index );
    free( elitist_archive_evolution_due );

    elitist_archive_size              = elitist_archive_size_new;
    elitist_archive                   = elitist_archive_new;
    elitist_archive_objective_values  = elitist_archive_objective_values_new;
    elitist_archive_constraint_values = elitist_archive_constraint_values_new;
    elitist_archive_from = elitist_archive_from_new;
    elitist_archive_chromosome_index = elitist_archive_chromosome_index_new;
    elitist_archive_evolution_due = elitist_archive_evolution_due_new;
}

/**
 * Returns 1 if index is in the indices array, 0 otherwise.
 */
short isInListOfIndices( int index, int *indices, int number_of_indices )
{
    int i;

    for( i = 0; i < number_of_indices; i++ )
        if( indices[i] == index )
        return( 1 );

    return( 0 );
}

/**
 * Adds a solution to the elitist archive.
 */
void addToElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, int solution_from, int chromosome_index)
{
    int      i, j, elitist_archive_capacity_new;
    char **elitist_archive_new;
    double **elitist_archive_objective_values_new;
	double *elitist_archive_constraint_values_new;
	int *elitist_archive_from_new;
    int *elitist_archive_chromosome_index_new;
    int *elitist_archive_evolution_due_new;

    if( elitist_archive_capacity == elitist_archive_size )
    {
        elitist_archive_capacity_new          = elitist_archive_capacity*2+1;
        elitist_archive_new                   = (char **) Malloc( elitist_archive_capacity_new*sizeof( char * ) );
        elitist_archive_objective_values_new  = (double **) Malloc( elitist_archive_capacity_new*sizeof( double * ) );
        elitist_archive_constraint_values_new = (double *) Malloc( elitist_archive_capacity_new*sizeof( double ) );
        elitist_archive_from_new = (int *) Malloc( elitist_archive_capacity_new*sizeof( int ) );
        elitist_archive_chromosome_index_new = (int *) Malloc( elitist_archive_capacity_new*sizeof( int ) );
        elitist_archive_evolution_due_new = (int *) Malloc( elitist_archive_capacity_new*sizeof( int ) );

        for( i = 0; i < elitist_archive_capacity_new; i++ )
        {
            elitist_archive_new[i]                    = (char *) Malloc( number_of_parameters*sizeof( char ) );
            elitist_archive_objective_values_new[i]   = (double *) Malloc( number_of_objectives*sizeof( double ) );
        }

        for( i = 0; i < elitist_archive_size; i++ )
        {
            for( j = 0; j < number_of_parameters; j++ )
                elitist_archive_new[i][j] = elitist_archive[i][j];
            for( j = 0; j < number_of_objectives; j++ )
                elitist_archive_objective_values_new[i][j] = elitist_archive_objective_values[i][j];
            elitist_archive_constraint_values_new[i] = elitist_archive_constraint_values[i];
            elitist_archive_from_new[i] = elitist_archive_from[i];
            elitist_archive_chromosome_index_new[i] = elitist_archive_chromosome_index[i];
            elitist_archive_evolution_due_new[i] = elitist_archive_evolution_due[i];
        }

        for( i = 0; i < elitist_archive_capacity; i++ )
        {
            free( elitist_archive[i] );
            free( elitist_archive_objective_values[i] );
        }
        free( elitist_archive );
        free( elitist_archive_objective_values );
        free( elitist_archive_constraint_values );
        free( elitist_archive_from );
        free( elitist_archive_chromosome_index );
        free(elitist_archive_evolution_due);

        elitist_archive_capacity          = elitist_archive_capacity_new;
        elitist_archive                   = elitist_archive_new;
        elitist_archive_objective_values  = elitist_archive_objective_values_new;
        elitist_archive_constraint_values = elitist_archive_constraint_values_new;
        elitist_archive_from = elitist_archive_from_new;
        elitist_archive_chromosome_index = elitist_archive_chromosome_index_new;
        elitist_archive_evolution_due = elitist_archive_evolution_due_new;
    }

    for( j = 0; j < number_of_parameters; j++ ){
        elitist_archive[elitist_archive_size][j] = solution[j];
    }
    for( j = 0; j < number_of_objectives; j++ ){
        elitist_archive_objective_values[elitist_archive_size][j] = solution_objective_values[j];
    }
    elitist_archive_constraint_values[elitist_archive_size] = solution_constraint_value; // Notice here //
    elitist_archive_from[elitist_archive_size] = solution_from;
    elitist_archive_chromosome_index[elitist_archive_size] = chromosome_index;
    elitist_archive_evolution_due[elitist_archive_size] = gen_now;

    elitist_archive_size++;
}
/**
 * Adapts the objective box discretization. If the numbre
 * of solutions in the elitist archive is too high or too low
 * compared to the population size, the objective box
 * discretization is adjusted accordingly. In doing so, the
 * entire elitist archive is first emptied and then refilled.
 */
void adaptObjectiveDiscretization( void )
{
    int    i, j, k, na, nb, nc, elitist_archive_size_target_lower_bound, elitist_archive_size_target_upper_bound;
    double low, high, *elitist_archive_objective_ranges;

    elitist_archive_size_target_lower_bound = (int) (0.75*elitist_archive_size_target);
    elitist_archive_size_target_upper_bound = (int) (1.25*elitist_archive_size_target);

    if( objective_discretization_in_effect && (elitist_archive_size < elitist_archive_size_target_lower_bound) )
        objective_discretization_in_effect = 0;

    if( elitist_archive_size > elitist_archive_size_target_upper_bound )
    {
        objective_discretization_in_effect = 1;

        elitist_archive_objective_ranges = (double *) Malloc( number_of_objectives*sizeof( double ) );
        for( j = 0; j < number_of_objectives; j++ )
        {
            low  = elitist_archive_objective_values[0][j];
            high = elitist_archive_objective_values[0][j];

            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( elitist_archive_objective_values[i][j] < low )
                    low = elitist_archive_objective_values[i][j];
                if( elitist_archive_objective_values[i][j] > high )
                    high = elitist_archive_objective_values[i][j];
            }

            elitist_archive_objective_ranges[j] = high - low;
        }

        na = 1;
        nb = (int) pow(2.0,25.0);
        
        for( k = 0; k < 25; k++ )
        {
            nc = (na + nb) / 2;
            for( i = 0; i < number_of_objectives; i++ )
                objective_discretization[i] = elitist_archive_objective_ranges[i]/((double) nc);

            /* Restore the original elitist archive after the first cycle in this loop */
            if( k > 0 )
            {
                elitist_archive_size = 0;
                for( i = 0; i < elitist_archive_copy_size; i++ )
                    addToElitistArchive( elitist_archive_copy[i], elitist_archive_copy_objective_values[i], elitist_archive_copy_constraint_values[i], elitist_archive_from[i], elitist_archive_chromosome_index[i] );
            }

            /* Copy the entire elitist archive */
            if( elitist_archive_copy != NULL )
            {
                for( i = 0; i < elitist_archive_copy_size; i++ )
                {
                    free( elitist_archive_copy[i] );
                    free( elitist_archive_copy_objective_values[i] );
                }
                free( elitist_archive_copy );
                free( elitist_archive_copy_objective_values );
                free( elitist_archive_copy_constraint_values );
            }

            elitist_archive_copy_size              = elitist_archive_size;
            elitist_archive_copy                   = (char **) Malloc( elitist_archive_copy_size*sizeof( char * ) );
            elitist_archive_copy_objective_values  = (double **) Malloc( elitist_archive_copy_size*sizeof( double * ) );
            elitist_archive_copy_constraint_values = (double *) Malloc( elitist_archive_copy_size*sizeof( double ) );
      
            for( i = 0; i < elitist_archive_copy_size; i++ )
            {
                elitist_archive_copy[i]                  = (char *) Malloc( number_of_parameters*sizeof( char ) );
                elitist_archive_copy_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
            }
            for( i = 0; i < elitist_archive_copy_size; i++ )
            {
                for( j = 0; j < number_of_parameters; j++ )
                    elitist_archive_copy[i][j] = elitist_archive[i][j];
                for( j = 0; j < number_of_objectives; j++ )
                    elitist_archive_copy_objective_values[i][j] = elitist_archive_objective_values[i][j];
                elitist_archive_copy_constraint_values[i] = elitist_archive_constraint_values[i];
            }

            /* Clear the elitist archive */
            elitist_archive_size = 0;

            /* Rebuild the elitist archive */
            for( i = 0; i < elitist_archive_copy_size; i++ )
                updateElitistArchive( elitist_archive_copy[i], elitist_archive_copy_objective_values[i], elitist_archive_copy_constraint_values[i], elitist_archive_from[i], elitist_archive_chromosome_index[i]);

            if( elitist_archive_size <= elitist_archive_size_target_lower_bound )
                na = nc;
            else
                nb = nc;
        }

        free( elitist_archive_objective_ranges );
    }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=- Solution Comparison -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
char betterFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x < constraint_value_y)
                result = TRUE;
        }
    }
    else
    {
        if(constraint_value_y > 0)
            result = TRUE;
        else
        {
            if(optimization[objective_index] == MINIMIZATION)
            {
                if(objective_value_x[objective_index] < objective_value_y[objective_index])
                    result = TRUE;
            }
            else if(optimization[objective_index] == MAXIMIZATION) 
            {
                if(objective_value_x[objective_index] > objective_value_y[objective_index])
                    result = TRUE;
            }
        }
    }

    return ( result );
}

char equalFitness(double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x == constraint_value_y)
                result = TRUE;
        }
    }
    else
    {
        if(constraint_value_y == 0)
        {
            if(objective_value_x[objective_index] == objective_value_y[objective_index])
                result = TRUE;
        }
    }

    return ( result );
}
/**
 * Returns 1 if x constraint-Pareto-dominates y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x Pareto dominates y
 */
short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if( constraint_value_x < constraint_value_y )       
                 result = TRUE;
        }
    }
    else /* x is feasible */
    {
        if( constraint_value_y > 0) /* x is feasible and y is not */
            result = TRUE;
        else /* Both are feasible */
            result = paretoDominates( objective_values_x, objective_values_y );
    }

    return( result );
}

short constraintWeaklyParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y  )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x  <= constraint_value_y )      
                result = TRUE;
        }
    }
    else /* x is feasible */
    {
        if( constraint_value_y > 0 ) /* x is feasible and y is not */
            result = TRUE;
        else /* Both are feasible */
            result = weaklyParetoDominates( objective_values_x, objective_values_y );
    }

    return( result );
}

/**
 * Returns 1 if x Pareto-dominates y, 0 otherwise.
 */
short paretoDominates( double *objective_values_x, double *objective_values_y )
{
    short strict;
    int   i, result;

    result = 1;
    strict = 0;

    for( i = 0; i < number_of_objectives; i++ )
    {
        if( fabs( objective_values_x[i] - objective_values_y[i] ) >= 0.00001 )
        {
            if(optimization[i] == MINIMIZATION)
            {
                if( objective_values_x[i] > objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
                if( objective_values_x[i] < objective_values_y[i] )
                    strict = 1;    
            }
            else if(optimization[i] == MAXIMIZATION)
            {
                if( objective_values_x[i] < objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
                if( objective_values_x[i] > objective_values_y[i] )
                    strict = 1;                    
            }
            
        }
    }

    if( strict == 0 && result == 1 )
        result = 0;

    return( result );
}

short weaklyParetoDominates( double *objective_values_x, double *objective_values_y )
{
    int   i, result;
    result = 1;

    for( i = 0; i < number_of_objectives; i++ )
    {
        if( fabs( objective_values_x[i] - objective_values_y[i] ) >= 0.00001 )
        {
            if(optimization[i] == MINIMIZATION)
            {
                if( objective_values_x[i] > objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
            }
            else if(optimization[i] == MAXIMIZATION)
            {
                if( objective_values_x[i] < objective_values_y[i] )
                {
                    result = 0;
                    break;
                }                
            }

        }
    }
    
    return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-= Linkage Tree Learning -==-=-=-=-=-=-=-=-=-=-=*/
/**
 * Learn the linkage for a cluster (subpopulation).
 */
void learnLinkageTree( int cluster_index )
{
    char   done;
    int    i, j, k, a, b, c, r0, r1, *indices, *order,
         lt_index, factor_size, **mpm_new, *mpm_new_number_of_indices, mpm_new_length,
        *NN_chain, NN_chain_length;
    double p, *cumulative_probabilities, **S_matrix, mul0, mul1;

    /* Compute joint entropy matrix */
    for( i = 0; i < number_of_parameters; i++ )
    {
        for( j = i+1; j < number_of_parameters; j++ )
        {
            indices                  = (int *) Malloc( 2*sizeof( int ) );
            indices[0]               = i;
            indices[1]               = j;
            cumulative_probabilities = estimateParametersForSingleBinaryMarginal( cluster_index, indices, 2, &factor_size );

            MI_matrix[i][j] = 0.0;
            for( k = 0; k < factor_size; k++ )
            {
                if( k == 0 )
                    p = cumulative_probabilities[k];
                else
                    p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
                if( p > 0 )
                    MI_matrix[i][j] += -p*log2(p);
            }

            MI_matrix[j][i] = MI_matrix[i][j];

            free( indices );
            free( cumulative_probabilities );
        }
        indices                  = (int *) Malloc( 1*sizeof( int ) );
        indices[0]               = i;
        cumulative_probabilities = estimateParametersForSingleBinaryMarginal( cluster_index, indices, 1, &factor_size );

        MI_matrix[i][i] = 0.0;
        for( k = 0; k < factor_size; k++ )
        {
            if( k == 0 )
                p = cumulative_probabilities[k];
            else
                p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
            if( p > 0 )
                MI_matrix[i][i] += -p*log2(p);
        }

        free( indices );
        free( cumulative_probabilities );
    }

    /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y) */
    for( i = 0; i < number_of_parameters; i++ )
        for( j = i+1; j < number_of_parameters; j++ )
        {
            MI_matrix[i][j] = MI_matrix[i][i] + MI_matrix[j][j] - MI_matrix[i][j];
            MI_matrix[j][i] = MI_matrix[i][j];
        }


    /* Initialize MPM to the univariate factorization */
    order                 = createRandomOrdering( number_of_parameters );
    mpm                   = (int **) Malloc( number_of_parameters*sizeof( int * ) );
    mpm_number_of_indices = (int *) Malloc( number_of_parameters*sizeof( int ) );
    mpm_length            = number_of_parameters;
    for( i = 0; i < number_of_parameters; i++ )
    {
        indices                  = (int *) Malloc( 1*sizeof( int ) );
        indices[0]               = order[i];
        mpm[i]                   = indices;
        mpm_number_of_indices[i] = 1;
    }
    free( order );

    /* Initialize LT to the initial MPM */
    if( lt[cluster_index] != NULL )
    {
        for( i = 0; i < lt_length[cluster_index]; i++ )
            free( lt[cluster_index][i] );
        free( lt[cluster_index] );
        free( lt_number_of_indices[cluster_index] );
    }
    lt[cluster_index]                   = (int **) Malloc( (number_of_parameters+number_of_parameters-1)*sizeof( int * ) );
    lt_number_of_indices[cluster_index] = (int *) Malloc( (number_of_parameters+number_of_parameters-1)*sizeof( int ) );
    lt_length[cluster_index]            = number_of_parameters+number_of_parameters-1;
    lt_index             = 0;
    for( i = 0; i < mpm_length; i++ )
    {
        lt[cluster_index][lt_index]                   = mpm[i];
        lt_number_of_indices[cluster_index][lt_index] = mpm_number_of_indices[i];
        lt_index++;
    }

    /* Initialize similarity matrix */
    S_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( i = 0; i < number_of_parameters; i++ )
        S_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    for( i = 0; i < mpm_length; i++ )
        for( j = 0; j < mpm_length; j++ )
            S_matrix[i][j] = MI_matrix[mpm[i][0]][mpm[j][0]];
    for( i = 0; i < mpm_length; i++ )
        S_matrix[i][i] = 0;

    NN_chain        = (int *) Malloc( (number_of_parameters+2)*sizeof( int ) );
    NN_chain_length = 0;
    done            = FALSE;
    while( done == FALSE )
    {
        if( NN_chain_length == 0 )
        {
            NN_chain[NN_chain_length] = randomInt( mpm_length );
            NN_chain_length++;
        }

        while( NN_chain_length < 3 )
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_length );
            NN_chain_length++;
        }

        while( NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1] )
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_length );
            if( ((S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length]] == S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
            NN_chain_length++;
        }
        r0 = NN_chain[NN_chain_length-2];
        r1 = NN_chain[NN_chain_length-1];
        if( r0 > r1 )
        {
            a  = r0;
            r0 = r1;
            r1 = a;
        }
        NN_chain_length -= 3;

        if( r1 < mpm_length ) // This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain
        {
            indices = (int *) Malloc( (mpm_number_of_indices[r0]+mpm_number_of_indices[r1])*sizeof( int ) );
  
            i = 0;
            for( j = 0; j < mpm_number_of_indices[r0]; j++ )
            {
                indices[i] = mpm[r0][j];
                i++;
            }
            for( j = 0; j < mpm_number_of_indices[r1]; j++ )
            {
                indices[i] = mpm[r1][j];
                i++;
            }
    
            lt[cluster_index][lt_index]                   = indices;
            lt_number_of_indices[cluster_index][lt_index] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
            lt_index++;
  
            mul0 = ((double) mpm_number_of_indices[r0])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
            mul1 = ((double) mpm_number_of_indices[r1])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
            for( i = 0; i < mpm_length; i++ )
            {
                if( (i != r0) && (i != r1) )
                {
                    S_matrix[i][r0] = mul0*S_matrix[i][r0] + mul1*S_matrix[i][r1];
                    S_matrix[r0][i] = S_matrix[i][r0];
                }
            }
  
            mpm_new                   = (int **) Malloc( (mpm_length-1)*sizeof( int * ) );
            mpm_new_number_of_indices = (int *) Malloc( (mpm_length-1)*sizeof( int ) );
            mpm_new_length            = mpm_length-1;
            for( i = 0; i < mpm_new_length; i++ )
            {
                mpm_new[i]                   = mpm[i];
                mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
            }
  
            mpm_new[r0]                   = indices;
            mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
            if( r1 < mpm_length-1 )
            {
                mpm_new[r1]                   = mpm[mpm_length-1];
                mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length-1];
  
                for( i = 0; i < r1; i++ )
                {
                    S_matrix[i][r1] = S_matrix[i][mpm_length-1];
                    S_matrix[r1][i] = S_matrix[i][r1];
                }
  
                for( j = r1+1; j < mpm_new_length; j++ )
                {
                    S_matrix[r1][j] = S_matrix[j][mpm_length-1];
                    S_matrix[j][r1] = S_matrix[r1][j];
                }
            }
  
            for( i = 0; i < NN_chain_length; i++ )
            {
                if( NN_chain[i] == mpm_length-1 )
                {
                    NN_chain[i] = r1;
                    break;
                }
            }
  
            free( mpm );
            free( mpm_number_of_indices );
            mpm                   = mpm_new;
            mpm_number_of_indices = mpm_new_number_of_indices;
            mpm_length            = mpm_new_length;
  
            if( mpm_length == 1 )
                done = TRUE;
        }
    }

    free( NN_chain );

    free( mpm_new );
    free( mpm_number_of_indices );

    for( i = 0; i < number_of_parameters; i++ )
        free( S_matrix[i] );
    free( S_matrix );
}

/**
 * Estimates the cumulative probability distribution of a
 * single binary marginal for a cluster (subpopulation).
 */
double *estimateParametersForSingleBinaryMarginal( int cluster_index, int *indices, int number_of_indices, int *factor_size )
{
    int     i, j, index, power_of_two;
    char *solution;
    double *result;

    *factor_size = (int) pow( 2, number_of_indices );
    result       = (double *) Malloc( (*factor_size)*sizeof( double ) );

    for( i = 0; i < (*factor_size); i++ )
        result[i] = 0.0;

    for( i = 0; i < population_cluster_sizes[cluster_index]; i++ ) 
    {
        index        = 0;
        power_of_two = 1;
        for( j = number_of_indices-1; j >= 0; j-- )
        {
            solution = population[population_indices_of_cluster_members[cluster_index][i]];
            index += (solution[indices[j]] == TRUE) ? power_of_two : 0;
            power_of_two *= 2;
        }

        result[index] += 1.0;
    }

    for( i = 0; i < (*factor_size); i++ )
        result[i] /= (double) population_cluster_sizes[cluster_index];

    for( i = 1; i < (*factor_size); i++ )
        result[i] += result[i-1];

    result[(*factor_size)-1] = 1.0;

    return( result );
}

/**
 * Determines nearest neighbour according to similarity values.
 */
int determineNearestNeighbour( int index, double **S_matrix, int mpm_length )
{
    int i, result;

    result = 0;
    if( result == index )
        result++;
    for( i = 1; i < mpm_length; i++ )
    {
//    if( (S_matrix[index][i] > S_matrix[index][result]) && (i != index) )
        if( ((S_matrix[index][i] > S_matrix[index][result]) || ((S_matrix[index][i] == S_matrix[index][result]) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index) )
        result = i;
    }

    return( result );
}

void printLTStructure( int cluster_index )
{
    int i, j;

    for( i = 0; i < lt_length[cluster_index]; i++ )
    {
        printf("[");
        for( j = 0; j < lt_number_of_indices[cluster_index][i]; j++ )
        {
            printf("%d",lt[cluster_index][i][j]);
            if( j < lt_number_of_indices[cluster_index][i]-1 )
                printf(" ");
        }
        printf("]\n");
    }
    printf("\n");
    fflush( stdout );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MO-GOMEA -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Performs initializations that are required before starting a run.
 */
void initialize()
{
    number_of_populations++;

    initializeMemory();

    for(int i = 0; i < population_size; i++){
        history.push_back(vector<int>({0}));
        // printf("init\n");
    }

    initializePopulationAndFitnessValues();

    computeObjectiveRanges();

}

/**
 * Initializes the memory.
 */
void initializeMemory( void )
{
    int i;

    objective_ranges         = (double *) Malloc( population_size*sizeof( double ) );
    population               = (char **) Malloc( population_size*sizeof( char * ) );
    objective_values         = (double **) Malloc( population_size*sizeof( double * ) );
    constraint_values        = (double *) Malloc( population_size*sizeof( double ) );
    
    for( i = 0; i < population_size; i++ )
    {
        population[i]        = (char *) Malloc( number_of_parameters*sizeof( char ) );
        objective_values[i]  = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }   

    t_NIS                    = 0;
    number_of_generations    = 0;
}
/**
 * Initializes the population and the objective values by randomly
 * generation n solutions.
 */
void initializePopulationAndFitnessValues()
{
    int i, j;
    for( i = 0; i < population_size; i++ )
    {
        for( j = 0; j < number_of_parameters; j++ )
            population[i][j] = (randomInt( 2 ) == 1) ? TRUE : FALSE;
        evaluateIndividual( population[i], objective_values[i],  &(constraint_values[i]), NOT_EXTREME_CLUSTER );
        updateElitistArchive( population[i], objective_values[i], constraint_values[i], INITIALIZE, i);    
    }
}
/**
 * Computes the ranges of all fitness values
 * of all solutions currently in the populations.
 */
void computeObjectiveRanges( void )
{
    int    i, j;
    double low, high;

    for( j = 0; j < number_of_objectives; j++ )
    {
        low  = objective_values[0][j];
        high = objective_values[0][j];

        for( i = 0; i < population_size; i++ )
        {
            if( objective_values[i][j] < low )
                low = objective_values[i][j];
            if( objective_values[i][j] > high )
                high = objective_values[i][j];
        }

        objective_ranges[j] = high - low;
    }
}

void learnLinkageOnCurrentPopulation()
{
    int i, j, k, size_of_one_cluster;

    initializeClusters();

    population_indices_of_cluster_members = clustering(objective_values, population_size, number_of_objectives, 
                            number_of_mixing_components, &size_of_one_cluster);
    population_cluster_sizes = (int*)Malloc(number_of_mixing_components*sizeof(int));
    for(k = 0; k < number_of_mixing_components; k++)    
        population_cluster_sizes[k] = size_of_one_cluster;

    // find extreme-region clusters
    determineExtremeClusters();

    // learn linkage tree for every cluster
    for( i = 0; i < number_of_mixing_components; i++ )
        learnLinkageTree( i );
}

int** clustering(double **objective_values_pool, int pool_size, int number_of_dimensions, 
                    int number_of_clusters, int *pool_cluster_size )
{
    int i, j, k, j_min, number_to_select,
        *pool_indices_of_leaders, *k_means_cluster_sizes, **pool_indices_of_cluster_members_k_means,
        **pool_indices_of_cluster_members, size_of_one_cluster;
    double distance, distance_smallest, epsilon,
            **objective_values_pool_scaled, **objective_means_scaled_new, *distances_to_cluster;
            
    if (number_of_clusters > 1)
        *pool_cluster_size   = (2*pool_size)/number_of_clusters;
    else
    {
        *pool_cluster_size   = pool_size;
        pool_indices_of_cluster_members = (int**)Malloc(number_of_clusters * sizeof(int*));
        pool_indices_of_cluster_members[0] = (int*)Malloc(pool_size * sizeof(int));
        for(i = 0; i < pool_size; i++)
            pool_indices_of_cluster_members[0][i] = i;
        return (pool_indices_of_cluster_members);
    }

    size_of_one_cluster  = *pool_cluster_size;

    /* Determine the leaders */
    objective_values_pool_scaled = (double **) Malloc( pool_size*sizeof( double * ) );
    for( i = 0; i < pool_size; i++ )
        objective_values_pool_scaled[i] = (double *) Malloc( number_of_dimensions*sizeof( double ) );
    for( i = 0; i < pool_size; i++ )
        for( j = 0; j < number_of_dimensions; j++ )
            objective_values_pool_scaled[i][j] = objective_values_pool[i][j]/objective_ranges[j];

    /* Heuristically find k far-apart leaders */
    number_to_select             = number_of_clusters;
    pool_indices_of_leaders = greedyScatteredSubsetSelection( objective_values_pool_scaled, pool_size, number_of_dimensions, number_to_select );

    for( i = 0; i < number_of_clusters; i++ )
        for( j = 0; j < number_of_dimensions; j++ )
            objective_means_scaled[i][j] = objective_values_pool[pool_indices_of_leaders[i]][j]/objective_ranges[j];

    /* Perform k-means clustering with leaders as initial mean guesses */
    objective_means_scaled_new = (double **) Malloc( number_of_clusters*sizeof( double * ) );
    for( i = 0; i < number_of_clusters; i++ )
        objective_means_scaled_new[i] = (double *) Malloc( number_of_dimensions*sizeof( double ) );

    pool_indices_of_cluster_members_k_means = (int **) Malloc( number_of_clusters*sizeof( int * ) );
    for( i = 0; i < number_of_clusters; i++ )
        pool_indices_of_cluster_members_k_means[i] = (int *) Malloc( pool_size*sizeof( int ) );

    k_means_cluster_sizes = (int *) Malloc( number_of_clusters*sizeof( int ) );

    epsilon = 1e+308;
    while( epsilon > 1e-10 )
    {
        for( j = 0; j < number_of_clusters; j++ )
        {
            k_means_cluster_sizes[j] = 0;
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j][k] = 0.0;
        }

        for( i = 0; i < pool_size; i++ )
        {
            j_min             = -1;
            distance_smallest = -1;
            for( j = 0; j < number_of_clusters; j++ )
            {
                distance = distanceEuclidean( objective_values_pool_scaled[i], objective_means_scaled[j], number_of_dimensions );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                {
                    j_min             = j;
                    distance_smallest = distance;
                }
            }
            pool_indices_of_cluster_members_k_means[j_min][k_means_cluster_sizes[j_min]] = i;
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j_min][k] += objective_values_pool_scaled[i][k];
            k_means_cluster_sizes[j_min]++;
        }

        for( j = 0; j < number_of_clusters; j++ )
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j][k] /= (double) k_means_cluster_sizes[j];

        epsilon = 0;
        for( j = 0; j < number_of_clusters; j++ )
        {
            epsilon += distanceEuclidean( objective_means_scaled[j], objective_means_scaled_new[j], number_of_dimensions );
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled[j][k] = objective_means_scaled_new[j][k];
        }
    }

    /* Shrink or grow the result of k-means clustering to get the final equal-sized clusters */
    pool_indices_of_cluster_members = (int**)Malloc(number_of_clusters * sizeof(int*));
    distances_to_cluster = (double *) Malloc( pool_size*sizeof( double ) );
    for( i = 0; i < number_of_clusters; i++ )
    {
        for( j = 0; j < pool_size; j++ )
            distances_to_cluster[j] = distanceEuclidean( objective_values_pool_scaled[j], objective_means_scaled[i], number_of_dimensions );

        for( j = 0; j < k_means_cluster_sizes[i]; j++ )
            distances_to_cluster[pool_indices_of_cluster_members_k_means[i][j]] = 0;

        pool_indices_of_cluster_members[i]          = mergeSort( distances_to_cluster, pool_size );
    }

    // Re-calculate clusters' means
    for( i = 0; i < number_of_clusters; i++)
    {
        for (j = 0; j < number_of_dimensions; j++)
            objective_means_scaled[i][j] = 0.0;

        for (j = 0; j < size_of_one_cluster; j++)
        {
            for( k = 0; k < number_of_dimensions; k++)
                objective_means_scaled[i][k] +=
                    objective_values_pool_scaled[pool_indices_of_cluster_members[i][j]][k];
        }

        for (j = 0; j < number_of_dimensions; j++)
        {
            objective_means_scaled[i][j] /= (double) size_of_one_cluster;
        }
    }

    free( distances_to_cluster );
    free( k_means_cluster_sizes );
    for( i = 0; i < number_of_clusters; i++ )
        free( pool_indices_of_cluster_members_k_means[i] );
    free( pool_indices_of_cluster_members_k_means );
    for( i = 0; i < number_of_clusters; i++ )
        free( objective_means_scaled_new[i] );
    free( objective_means_scaled_new );
    for( i = 0; i < pool_size; i++ )
        free( objective_values_pool_scaled[i] );
    free( objective_values_pool_scaled );
    free( pool_indices_of_leaders );   

    return (pool_indices_of_cluster_members);
}
/**
 * Selects n points from a set of points. A
 * greedy heuristic is used to find a good
 * scattering of the selected points. First,
 * a point is selected with a maximum value
 * in a randomly selected dimension. The
 * remaining points are selected iteratively.
 * In each iteration, the point selected is
 * the one that maximizes the minimal distance
 * to the points selected so far.
 */
int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select )
{
    int     i, index_of_farthest, random_dimension_index, number_selected_so_far,
            *indices_left, *result;
    double *nn_distances, distance_of_farthest, value;

    if( number_to_select > number_of_points )
    {
        printf("\n");
        printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %d.", number_to_select, number_of_points);
        printf("\n\n");

        exit( 0 );
    }

    result = (int *) Malloc( number_to_select*sizeof( int ) );

    indices_left = (int *) Malloc( number_of_points*sizeof( int ) );
    for( i = 0; i < number_of_points; i++ )
        indices_left[i] = i;

    /* Find the first point: maximum value in a randomly chosen dimension */
    random_dimension_index = randomInt( number_of_dimensions );

    index_of_farthest    = 0;
    distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
    for( i = 1; i < number_of_points; i++ )
    {
        if( points[indices_left[i]][random_dimension_index] > distance_of_farthest )
        {
            index_of_farthest    = i;
            distance_of_farthest = points[indices_left[i]][random_dimension_index];
        }
    }

    number_selected_so_far          = 0;
    result[number_selected_so_far]  = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
    number_selected_so_far++;

    /* Then select the rest of the solutions: maximum minimum
     * (i.e. nearest-neighbour) distance to so-far selected points */
    nn_distances = (double *) Malloc( number_of_points*sizeof( double ) );
    for( i = 0; i < number_of_points-number_selected_so_far; i++ )
        nn_distances[i] = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );

    while( number_selected_so_far < number_to_select )
    {
        index_of_farthest    = 0;
        distance_of_farthest = nn_distances[0];
        for( i = 1; i < number_of_points-number_selected_so_far; i++ )
        {
            if( nn_distances[i] > distance_of_farthest )
            {
                index_of_farthest    = i;
                distance_of_farthest = nn_distances[i];
            }
        }

        result[number_selected_so_far]  = indices_left[index_of_farthest];
        indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
        nn_distances[index_of_farthest] = nn_distances[number_of_points-number_selected_so_far-1];
        number_selected_so_far++;

        for( i = 0; i < number_of_points-number_selected_so_far; i++ )
        {
            value = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );
            if( value < nn_distances[i] )
                nn_distances[i] = value;
        }
    }

    free( nn_distances );
    free( indices_left );
    return( result );
}

void determineExtremeClusters()
{
    int i,j, index_best, a,b,c, *order;
    // find extreme clusters
    order = createRandomOrdering(number_of_objectives);
        
    for (i = 0; i < number_of_mixing_components; i++)
        which_extreme[i] = -1;  // not extreme cluster
    
    if(number_of_mixing_components > 1)
    {
        for (i = 0; i < number_of_objectives; i++)
        {
            index_best = -1;
        
            for (j = 0; j < number_of_mixing_components; j++)
            {
                if(optimization[order[i]] == MINIMIZATION)
                {
                    if( ((index_best == -1) || (objective_means_scaled[j][order[i]] < objective_means_scaled[index_best][order[i]]) )&&
                        (which_extreme[j] == -1) )
                        index_best = j;
                }
                else if(optimization[order[i]] == MAXIMIZATION)
                {
                    if( ((index_best == -1) || (objective_means_scaled[j][order[i]] > objective_means_scaled[index_best][order[i]]) )&&
                        (which_extreme[j] == -1) )
                        index_best = j;
                }
            }
            which_extreme[index_best] = order[i];
        }
    }

    free(order);
}

void initializeClusters()
{
    int i;
    lt                            = (int ***) Malloc( number_of_mixing_components*sizeof( int ** ) );
    lt_length                     = (int *) Malloc( number_of_mixing_components*sizeof( int ) );
    lt_number_of_indices          = (int **) Malloc( number_of_mixing_components*sizeof( int *) );
    for( i = 0; i < number_of_mixing_components; i++)
    {
        lt[i]                     = NULL;
        lt_number_of_indices[i]   = NULL;
        lt_length[i]              = 0;
    }

    which_extreme                 = (int*)Malloc(number_of_mixing_components*sizeof(int));

    objective_means_scaled        = (double **) Malloc( number_of_mixing_components*sizeof( double * ) );    
    for( i = 0; i < number_of_mixing_components; i++ )
        objective_means_scaled[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
}

void ezilaitiniClusters()
{
    int i, j;

    if(lt == NULL)
        return;

    for( i = 0; i < number_of_mixing_components; i++ )
    {
        if( lt[i] != NULL )
        {
            for( j = 0; j < lt_length[i]; j++ )
                free( lt[i][j] );
            free( lt[i] );
            free( lt_number_of_indices[i] );
        }
    }

    free( lt ); lt = NULL;
    free( lt_length );
    free( lt_number_of_indices );

    free(which_extreme);

    for(i = 0; i < number_of_mixing_components; i++)
        free(objective_means_scaled[i]);
    free( objective_means_scaled );
}

void improveCurrentPopulation( void )
{
    int     i, j, k, j_min, cluster_index, objective_index, number_of_cluster,
            *sum_cluster, *clusters ;
    double *objective_values_scaled,
          distance, distance_smallest;

    offspring_size                  = population_size;
    offspring                       = (char**)Malloc(offspring_size*sizeof(char*));
    objective_values_offspring      = (double**)Malloc(offspring_size*sizeof(double*));
    constraint_values_offspring     = (double*)Malloc(offspring_size*sizeof(double));
    
    for(i = 0; i < offspring_size; i++)
    {
        offspring[i]                = (char*)Malloc(number_of_parameters*sizeof(char));
        objective_values_offspring[i]  = (double*)Malloc(number_of_objectives*sizeof(double));
    }

    objective_values_scaled = (double *) Malloc( number_of_objectives*sizeof( double ) );
    sum_cluster = (int*)Malloc(number_of_mixing_components*sizeof(int));

    for(i = 0; i < number_of_mixing_components; i++)
        sum_cluster[i] = 0;

    elitist_archive_front_changed = FALSE;
    for( i = 0; i < population_size; i++ )
    {
        number_of_cluster = 0;
        clusters = (int*)Malloc(number_of_mixing_components*sizeof(int));
        for(j = 0; j < number_of_mixing_components; j++)
        {
            for (k = 0; k < population_cluster_sizes[j]; k++)
            {
                if(population_indices_of_cluster_members[j][k] == i)
                {
                    clusters[number_of_cluster] = j;
                    number_of_cluster++;
                    break;
                }
            }
        }
        if(number_of_cluster > 0)
            cluster_index = clusters[randomInt(number_of_cluster)];
        else
        {
            for( j = 0; j < number_of_objectives; j++ )
                objective_values_scaled[j] = objective_values[i][j]/objective_ranges[j];

            distance_smallest = -1;
            j_min = -1;
            for( j = 0; j < number_of_mixing_components; j++ )
            {
                distance = distanceEuclidean( objective_values_scaled, objective_means_scaled[j], number_of_objectives );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                {
                    j_min = j;
                    distance_smallest  = distance;
                }
            
            }

            cluster_index = j_min;
        }

        int population_base = 0;
        for(int _i = 0; _i < population_id; _i++){
            population_base += array_of_population_sizes[_i];
        }

        int from_which;
        sum_cluster[cluster_index]++;
        if(which_extreme[cluster_index] == -1)
        {
            bool doMOGOM = false;
            int elitist_num = elitist_num_history.size();
            
            if(elitist_num >= 3){
                if(elitist_num_history[elitist_num-1] == elitist_num_history[elitist_num-2]
                 && elitist_num_history[elitist_num-2] == elitist_num_history[elitist_num-3]
                 ){
                    doMOGOM = true;
                }
            }

            // if (doMOGOM)
            //     printf("in not stop scheme\n");

            // if(doMOGOM){
            //     performMultiObjectiveGenepoolOptimalMixing( cluster_index, population[i], objective_values[i], constraint_values[i],
            //                     offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), &from_which, population_base+i);
            // }
            // else{
                performForceImprovement(cluster_index, population[i], objective_values[i], constraint_values[i],
                                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), &from_which, population_base+i, doMOGOM);
            // }
            // performCheatingOM(cluster_index, population[i], objective_values[i], constraint_values[i],
            //                     offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), &from_which, population_base+i);
          
        }
        else
        {
            objective_index = which_extreme[cluster_index];
            performSingleObjectiveGenepoolOptimalMixing(cluster_index, objective_index, population[i], objective_values[i], constraint_values[i], 
                                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), &from_which, population_base+i);
        }

        history[population_base+i].push_back(from_which);
        free(clusters);
    }

    free( objective_values_scaled ); free( sum_cluster );

    if(!elitist_archive_front_changed)
        t_NIS++;
    else
        t_NIS = 0;
}

void copyValuesFromDonorToOffspring(char *solution, char *donor, int cluster_index, int linkage_group_index)
{
    int i, parameter_index;
    for (i = 0; i < lt_number_of_indices[cluster_index][linkage_group_index]; i++)
    {
        parameter_index = lt[cluster_index][linkage_group_index][i];
        solution[parameter_index] = donor[parameter_index];    
    }
}

void copyValuesFromDonorToOffspringComplement(char *solution, char *donor, int cluster_index, int linkage_group_index)
{
    int i, parameter_index;
    for (i = 0; i < lt_number_of_indices[cluster_index][linkage_group_index]; i++)
    {
        parameter_index = lt[cluster_index][linkage_group_index][i];
        solution[parameter_index] = (donor[parameter_index]==0?1:0);    
    }
}

void copyFromAToB(char *solution_a, double *obj_a, double con_a, char *solution_b, double *obj_b, double *con_b)
{
    int i;
    for (i = 0; i < number_of_parameters; i++)
        solution_b[i] = solution_a[i];
    for (i = 0; i < number_of_objectives; i++)
        obj_b[i] = obj_a[i];
    *con_b = con_a;
}

void mutateSolution(char *solution, int lt_factor_index, int cluster_index)
{
    double mutation_rate, prob;
    int i, parameter_index;

    if(use_pre_mutation == FALSE && use_pre_adaptive_mutation == FALSE)
        return;

    mutation_rate = 0.0;
    if(use_pre_mutation == TRUE)
        mutation_rate = 1.0/((double)number_of_parameters);
    else if(use_pre_adaptive_mutation == TRUE)
        mutation_rate = 1.0/((double)lt_number_of_indices[cluster_index][lt_factor_index]);

    
    for(i = 0; i < lt_number_of_indices[cluster_index][lt_factor_index]; i++)
    {
        prob = randomRealUniform01();
        if(prob < mutation_rate)
        {
            parameter_index = lt[cluster_index][lt_factor_index][i];
            if(solution[parameter_index] == 0) 
                solution[parameter_index] = 1;
            else
                solution[parameter_index] = 0;
        }
    }
    
}
/**
 * Multi-objective Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in a middle-region cluster.
 */
void performMultiObjectiveGenepoolOptimalMixing( int cluster_index, char *parent, double *parent_obj, double parent_con, 
                            char *result, double *obj,  double *con, int *from_which, int chromosome_index)
{
    total_mixing_times++;
    char   *backup, *donor, is_unchanged, changed, is_improved, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, index, donor_index, position_of_existed_member, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, result, obj, con);
    
    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup);

    number_of_linkage_sets = lt_length[cluster_index] - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);
    
    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        donor_index = randomInt( population_cluster_sizes[cluster_index] );

        donor = population[population_indices_of_cluster_members[cluster_index][donor_index]];


        copyValuesFromDonorToOffspring(result, donor, cluster_index, linkage_group_index);     
        mutateSolution(result, linkage_group_index, cluster_index);

        /* Check if the new intermediate solution is different from the previous state. */
        is_unchanged = TRUE;
        for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
        {
            if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
            {
                is_unchanged = FALSE;
                break;
            }
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;            
            evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, MIDDLE_GOM, chromosome_index, from_which);
            
            if (is_new_nondominated_point){
                if((objective_values[donor_index][0] - obj_backup[0] < 0) && (objective_values[donor_index][1] - obj_backup[1] < 0)){
                    both_small_cnt ++;
                }
                double angle = getAngle(objective_values[donor_index], obj_backup, obj);
                writeCurrentAngle(angle, false);
                // printf("middle GOM donor: (%f, %f), (%f, %f) -> (%f, %f)\n",objective_values[donor_index][0], objective_values[donor_index][1], obj_backup[0], obj_backup[1], obj[0], obj[1]);
                if (angle > 90){
                    anti_cnt ++;
                    // printf("angle is %f and it's to the anti way\n", angle);
                }
                else{
                    // printf("angle is %f and it's to the way\n", angle);
                    right_cnt ++;
                }
                // printf("GOM-> right: %d, anti: %d both small:%d, FI-> right: %d, anti: %d\n", right_cnt, anti_cnt,both_small_cnt, FI_right_cnt, FI_anti_cnt);
            }

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_improved = TRUE;
            
            if ( is_improved )
            {
                changed = TRUE;
                *from_which = MIDDLE_GOM;
                // writeCurrentMask(cluster_index, linkage_group_index, result, backup);
                copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);

                if(!checkInCluster(cluster_index, linkage_group_index, donor)){
                    in_cluster_time++;
                }
                total_cnt++;
            }
            
            else
                copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
        }
    }
    free(order);

    /* Forced Improvement */
    if (  (!changed) || (t_NIS > (1+floor(log10(population_size))))  )
    {
        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            total_FI_times++;

            donor_index = randomInt(elitist_archive_size);
            linkage_group_index = order[i];
            donor = elitist_archive[donor_index];
            
            copyValuesFromDonorToOffspring(result, elitist_archive[donor_index], cluster_index, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_index);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, MIDDLE_FI, chromosome_index, from_which);


                if (is_new_nondominated_point){
                    double angle = getAngle(elitist_archive_objective_values[donor_index], obj_backup, obj);
                    writeCurrentAngle(angle, true);

                    // printf("middle FI donor: (%f, %f), (%f, %f) -> (%f, %f)\n",elitist_archive_objective_values[donor_index][0], elitist_archive_objective_values[donor_index][1], obj_backup[0], obj_backup[1], obj[0], obj[1]);
                    if (angle > 90){
                        FI_anti_cnt ++;
                        // printf("angle is %f and it's to the anti way\n", angle);
                    }
                    else{
                        // printf("angle is %f and it's to the way\n", angle);
                        FI_right_cnt ++;
                    }
                    // printf("GOM-> right: %d, anti: %d, FI-> right: %d, anti: %d\n", right_cnt, anti_cnt, FI_right_cnt, FI_anti_cnt);
                }
                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    *from_which = 4;
                    total_FI_improve_times++;
                    copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);

                    if(!checkInCluster(cluster_index, linkage_group_index, donor)){
                        in_cluster_time++;
                    }
                    total_cnt++;
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
            }
        }
        free(order);

        if(!changed)
        {
            donor_index = randomInt( elitist_archive_size );

            copyFromAToB(elitist_archive[donor_index], elitist_archive_objective_values[donor_index], 
                    elitist_archive_constraint_values[donor_index], 
                    result, obj, con);
        }
    }

    free( backup ); free( obj_backup ); 
}
/**
 * Single-objective Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in an extreme-region cluster.
 */
void performSingleObjectiveGenepoolOptimalMixing( int cluster_index, int objective_index, 
                                char *parent, double *parent_obj, double parent_con,
                                char *result, double *obj, double *con, int *from_which, int chromosome_index )
{
    char   *backup, *donor, *elitist_copy, is_unchanged, changed, is_improved, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, index, donor_index, number_of_linkage_sets, linkage_group_index, *order;
    double  *obj_backup, con_backup;
    total_mixing_times++;
    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, result, obj, con);

    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup);

    number_of_linkage_sets = lt_length[cluster_index] - 1; /* Remove root from the linkage tree. */
    
    order = createRandomOrdering(number_of_linkage_sets);

    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];
        donor_index = randomInt( population_cluster_sizes[cluster_index] );
        
        donor = population[population_indices_of_cluster_members[cluster_index][donor_index]];
        copyValuesFromDonorToOffspring(result, donor, cluster_index, linkage_group_index);        
        mutateSolution(result, linkage_group_index, cluster_index);

        /* Check if the new intermediate solution is different from the previous state. */
        is_unchanged = TRUE;
        for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
        {
            if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
            {
                is_unchanged = FALSE;
                break;
            }
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            evaluateIndividual(result, obj, con, objective_index);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, EXTREME_GOM, chromosome_index, from_which);

            if(betterFitness(obj, *con, obj_backup, con_backup, objective_index) ||
                equalFitness(obj, *con, obj_backup, con_backup, objective_index) )
                is_improved = TRUE;
            
            if ( is_improved )
            {
                changed = TRUE;
                *from_which = 1;
                copyFromAToB(result, obj, *con,  backup, obj_backup, &con_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, result, obj, con);    
        }
    }
    free(order);

    elitist_copy = (char*)Malloc(number_of_parameters*sizeof(char));
    /* Forced Improvement*/
    if ( (!changed) || ( t_NIS > (1+floor(log10(population_size))) ) ) 
    {
        changed = FALSE;
        /* Find in the archive the solution having the best value in the corresponding objective. */
        donor_index = 0;
        for (j= 0; j < elitist_archive_size; j++)
        {
            if(optimization[objective_index] == MINIMIZATION)
            {
                if(elitist_archive_objective_values[j][objective_index] < elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;
            }
            else if(optimization[objective_index] == MAXIMIZATION)
            {
                if(elitist_archive_objective_values[j][objective_index] > elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;   
            }
        }   

        for (j = 0; j < number_of_parameters; j++)
            elitist_copy[j] = elitist_archive[donor_index][j];

        /* Perform Gene-pool Optimal Mixing with the single-objective best-found solution as the donor. */
        order = createRandomOrdering(number_of_linkage_sets);
        for( i = 0; i < number_of_linkage_sets; i++ )
        {
            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, elitist_copy, cluster_index, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_index);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_index][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_index][linkage_group_index][j]] != result[lt[cluster_index][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;
                evaluateIndividual(result, obj, con, objective_index);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, &is_new_nondominated_point, &is_dominated_by_archive, EXTREME_FI, chromosome_index, from_which);

                /* Check if strict improvement in the corresponding objective. */
                if(betterFitness(obj, *con, obj_backup, con_backup, objective_index) )
                    is_improved = TRUE;

                if (is_improved == TRUE)
                {
                    changed = TRUE;
                    *from_which = 3;
                    copyFromAToB(result, obj, *con, backup, obj_backup, &con_backup );
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, result, obj, con);
            }
        }
        free(order);

        if(!changed)
        {
            // printf("weird FI\n");
            donor_index = 0;
            for (j= 0; j < elitist_archive_size; j++)
            {
                if(optimization[objective_index] == MINIMIZATION)
                {
                    if(elitist_archive_objective_values[j][objective_index] < elitist_archive_objective_values[donor_index][objective_index])
                        donor_index = j;
                }
                else if(optimization[objective_index] == MAXIMIZATION)
                {
                    if(elitist_archive_objective_values[j][objective_index] > elitist_archive_objective_values[donor_index][objective_index])
                        donor_index = j;   
                }
            }

            copyFromAToB(elitist_archive[donor_index], elitist_archive_objective_values[donor_index], 
                    elitist_archive_constraint_values[donor_index], 
                    result, obj, con);
        }
    }
    
    free( backup ); free( obj_backup ); free( elitist_copy );
}
/**
 * Determines the solutions that finally survive the generation (offspring only).
 */
void selectFinalSurvivors()
{
    int i, j;

    for( i = 0; i < population_size; i++ )
    {
        for( j = 0; j < number_of_parameters; j++ )
            population[i][j] = offspring[i][j];
        for( j = 0; j < number_of_objectives; j++)
            objective_values[i][j]  = objective_values_offspring[i][j];
        constraint_values[i] = constraint_values_offspring[i];
    }
}

void freeAuxiliaryPopulations()
{
    int i, k;

    if(population_indices_of_cluster_members != NULL)
    {
        for(k = 0; k < number_of_mixing_components; k++)
            free(population_indices_of_cluster_members[k]);
        free(population_indices_of_cluster_members);
        population_indices_of_cluster_members = NULL;
        free(population_cluster_sizes);
    }

    if(offspring != NULL)
    {
        for(i = 0; i < offspring_size; i++)
        {
            free(offspring[i]);
            free(objective_values_offspring[i]);
        }
        free(offspring);
        free(objective_values_offspring);
        free(constraint_values_offspring);
        offspring = NULL;
    }
    
    ezilaitiniClusters();
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Parameter-free Mechanism -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void initializeMemoryForArrayOfPopulations()
{
    int i;
    maximum_number_of_populations      = 20;
    // maximum_number_of_populations      = 1;

    array_of_populations                = (char***)Malloc(maximum_number_of_populations*sizeof(char**));
    array_of_objective_values           = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    array_of_constraint_values          = (double**)Malloc(maximum_number_of_populations*sizeof(double*));
    array_of_objective_ranges           = (double**)Malloc(maximum_number_of_populations*sizeof(double));

    array_of_t_NIS                      = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_number_of_generations                = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    for(i = 0; i < maximum_number_of_populations; i++)
    {
        array_of_number_of_generations[i]         = 0;
        array_of_t_NIS[i]               = 0;
    }

    array_of_number_of_evaluations_per_population = (long*)Malloc(maximum_number_of_populations*sizeof(long));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_number_of_evaluations_per_population[i] = 0;

    /* Popupulation-sizing free scheme. */
    array_of_population_sizes           = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_population_sizes[0]        = smallest_population_size;
    for(i = 1; i < maximum_number_of_populations; i++)
        array_of_population_sizes[i]    = array_of_population_sizes[i-1]*2;

    /* Number-of-clusters parameter-free scheme. */
    array_of_number_of_clusters         = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_number_of_clusters[0]      = number_of_objectives + 1;
    // array_of_number_of_clusters[0]      = 20;
    for(i = 1; i < maximum_number_of_populations; i++)
        array_of_number_of_clusters[i]  = array_of_number_of_clusters[i-1] + 1;
}

void putInitializedPopulationIntoArray()
{
    array_of_objective_ranges[population_id]    = objective_ranges;
    array_of_populations[population_id]         = population;
    array_of_objective_values[population_id]    = objective_values;
    array_of_constraint_values[population_id]   = constraint_values;
    array_of_t_NIS[population_id]               = 0;
}

void assignPointersToCorrespondingPopulation()
{
    population                  = array_of_populations[population_id];
    objective_values            = array_of_objective_values[population_id];
    constraint_values           = array_of_constraint_values[population_id];
    population_size             = array_of_population_sizes[population_id];
    objective_ranges            = array_of_objective_ranges[population_id];
    t_NIS                       = array_of_t_NIS[population_id];
    number_of_generations       = array_of_number_of_generations[population_id];
    number_of_mixing_components = array_of_number_of_clusters[population_id];
}

void ezilaitiniMemoryOfCorrespondingPopulation()
{
    int i;

    for( i = 0; i < population_size; i++ )
    {
        free( population[i] );
        free( objective_values[i] );
    }
    free( population );
    free( objective_values );
    free( constraint_values );
    free( objective_ranges );
}

void ezilaitiniArrayOfPopulation()
{
    int i;
    for(i = 0; i < number_of_populations; i++)
    {
        population_id = i;
        assignPointersToCorrespondingPopulation();
        ezilaitiniMemoryOfCorrespondingPopulation();
    }
    free(array_of_populations);
    free(array_of_objective_values);
    free(array_of_constraint_values);
    free(array_of_population_sizes);
    free(array_of_objective_ranges);
    free(array_of_t_NIS);
    free(array_of_number_of_generations);
    free(array_of_number_of_evaluations_per_population);
    free(array_of_number_of_clusters);
}
/**
 * Schedule the run of multiple populations.
 */
void schedule_runMultiplePop_clusterPop_learnPop_improvePop()
{
    int i;
    smallest_population_size = 8;
    // smallest_population_size = 7000;

    initializeMemoryForArrayOfPopulations();
    initializeArrayOfParetoFronts();
    while( !checkTerminationCondition() )
    {
        population_id = 0;
        do
        {
            gen_now = array_of_number_of_generations[population_id];
            // printf("gen_now = %d\n", gen_now);
            if(array_of_number_of_generations[population_id] == 0)
            {
                population_size = array_of_population_sizes[population_id];
                number_of_mixing_components = array_of_number_of_clusters[population_id];

                initialize();

                putInitializedPopulationIntoArray();

                if(stop_population_when_front_is_covered)
                {
                    updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                    checkWhichSmallerPopulationsNeedToStop();
                }

                writeGenerationalStatistics();
            }
            else if(array_of_population_statuses[population_id] == TRUE)
            {
                assignPointersToCorrespondingPopulation();

                learnLinkageOnCurrentPopulation();

                improveCurrentPopulation();
                // printf("FI/total: %d/%d\n", total_FI_times, total_mixing_times);

                selectFinalSurvivors();

                computeObjectiveRanges();

                adaptObjectiveDiscretization();

                array_of_t_NIS[population_id] = t_NIS;

                if(stop_population_when_front_is_covered)
                {
                    updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                    checkWhichSmallerPopulationsNeedToStop();
                }

                writeGenerationalStatistics();
            }
            
            array_of_number_of_generations[population_id]++;
            if(use_print_progress_to_screen)
                printf("%d ", array_of_number_of_generations[population_id]);
            population_id++;
            // writeCurrentHistory();
            // for(int _i =0; _i < elitist_archive_size;_i++){
            //     int idx = elitist_archive_chromosome_index[_i];
            //     int due = elitist_archive_evolution_due[_i];
            //     printf("%d, %d\n", idx, due);
            //     for(int _j = 0; _j <= due; _j++){
            //         printf("%d->",history[idx][_j]);
            //     }
            //     printf("\n");
            // }
            if(checkTerminationCondition() == TRUE)
                break;
        } while(array_of_number_of_generations[population_id-1] % generation_base == 0);
        if(use_print_progress_to_screen)
            printf(":   %d\n", number_of_evaluations);
        
        // printf("FI/total: %d/%d\n", total_FI_times, total_mixing_times);
        
    }
    if(use_print_progress_to_screen)
    {
        printf("Population Status:\n");
        for(i=0; i < number_of_populations; i++)
            printf("Pop %d: %d\n", ((int)(pow(2,i)))*smallest_population_size, array_of_population_statuses[i]);
    }
    printf("FI improve/FI/total: %ld/%ld/%ld\n",total_FI_improve_times, total_FI_times, total_mixing_times);
    printf("total complement success: %ld\n", total_complement_count);
    printf("Masks in cluster/ total masks : %ld/%ld\n",in_cluster_time, total_cnt);
    writeCurrentNotInClusterRatio(in_cluster_time, total_cnt);
    logNumberOfEvaluationsAtVTR();
    for(i =0; i < 5; i++){
        printf("%d\n",array_of_number_of_generations[i]);
    }
    // writeCurrentElitistArchive( TRUE );
    // writeCurrentElitistArchiveFrom( TRUE );
    writeCurrentHistory();
    ezilaitiniArrayOfPopulation();
    ezilaitiniArrayOfParetoFronts();
}

void schedule()
{   
    schedule_runMultiplePop_clusterPop_learnPop_improvePop();
}

/*---------------------Section Stop Smaller Populations -----------------------------------*/
void initializeArrayOfParetoFronts()
{
    int i;
    
    array_of_population_statuses                    = (char*)Malloc(maximum_number_of_populations*sizeof(char));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_population_statuses[i] = TRUE;
    
    array_of_Pareto_front_size_of_each_population   = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_Pareto_front_size_of_each_population[i] = 0;

    array_of_Pareto_front_of_each_population        = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_Pareto_front_of_each_population[i] = NULL;
}

void ezilaitiniArrayOfParetoFronts()
{
    int i, j;

    FILE *file;
    file = fopen("population_status.dat", "w");
    for(i = 0; i < number_of_populations; i++)
    {
        fprintf(file, "Pop %d: %d\n", ((int)(pow(2,i)))*smallest_population_size, array_of_population_statuses[i]);
    }
    fclose(file);
    for(i = 0; i < maximum_number_of_populations; i++)
    {
        if(array_of_Pareto_front_size_of_each_population[i] > 0)
        {
            for(j = 0; j < array_of_Pareto_front_size_of_each_population[i]; j++)
                free(array_of_Pareto_front_of_each_population[i][j]);
            free(array_of_Pareto_front_of_each_population[i]);
        }
    }
    free(array_of_Pareto_front_of_each_population);
    free(array_of_Pareto_front_size_of_each_population);
    free(array_of_population_statuses);
}

char checkParetoFrontCover(int pop_index_1, int pop_index_2)
{
    int i, j, count;
    count = 0;
    
    for(i = 0; i < array_of_Pareto_front_size_of_each_population[pop_index_2]; i++)
    {
        for(j = 0; j < array_of_Pareto_front_size_of_each_population[pop_index_1]; j++)
            if((constraintParetoDominates(array_of_Pareto_front_of_each_population[pop_index_1][j], 0, 
                array_of_Pareto_front_of_each_population[pop_index_2][i], 0) == TRUE) ||
                sameObjectiveBox(array_of_Pareto_front_of_each_population[pop_index_1][j], array_of_Pareto_front_of_each_population[pop_index_2][i]) == TRUE)
        {
            count++;
            break;
        }
    }
    // Check if all points in front 2 are dominated by or exist in front 1
    if(count == array_of_Pareto_front_size_of_each_population[pop_index_2])
        return TRUE;
    return FALSE;
}

void checkWhichSmallerPopulationsNeedToStop()
{
    int i;
    for(i = population_id - 1; i >= 0; i--)
    {
        if(array_of_population_statuses[i] == FALSE)
            continue;
        if(checkParetoFrontCover(population_id, i) == TRUE)
            array_of_population_statuses[i] = FALSE;
    }
}

void updateParetoFrontForCurrentPopulation(double **objective_values_pop, double *constraint_values_pop, int pop_size)
{
    int i, j, index, rank0_size;
    char *isDominated;
    isDominated = (char*)Malloc(pop_size*sizeof(char));
    for(i = 0; i < pop_size; i++)
        isDominated[i] = FALSE;
    for (i = 0; i < pop_size; i++)
    {
        if(isDominated[i] == TRUE)
            continue;
        for(j = i+1; j < pop_size; j++)
        {
            if(isDominated[j] == TRUE)
                continue;
            if(constraintParetoDominates(objective_values_pop[i], constraint_values_pop[i], objective_values_pop[j],constraint_values_pop[j]) == TRUE)
                isDominated[j]=TRUE;
            else if(constraintParetoDominates(objective_values_pop[j], constraint_values_pop[j], objective_values_pop[i],constraint_values_pop[i]) == TRUE)
            {
                isDominated[i]=TRUE;
                break;
            }
        }
    }

    rank0_size = 0;
    for(i = 0; i < pop_size; i++)
        if(isDominated[i]==FALSE)
            rank0_size++;

    if(array_of_Pareto_front_size_of_each_population[population_id] > 0)
    {
        for(i = 0; i < array_of_Pareto_front_size_of_each_population[population_id]; i++)
        {
            free(array_of_Pareto_front_of_each_population[population_id][i]);
        }
        free(array_of_Pareto_front_of_each_population[population_id]);        
    }

    array_of_Pareto_front_of_each_population[population_id] = (double**)Malloc(rank0_size*sizeof(double*));
    for(i = 0; i < rank0_size; i++)
        array_of_Pareto_front_of_each_population[population_id][i] = (double*)Malloc(number_of_objectives*sizeof(double));
    array_of_Pareto_front_size_of_each_population[population_id] = rank0_size;

    index = 0;
    for(i = 0; i < pop_size; i++)
    {
        if(isDominated[i] == TRUE)
            continue;
        for(j = 0; j < number_of_objectives; j++)
            array_of_Pareto_front_of_each_population[population_id][index][j] = objective_values_pop[i][j];
        index++;
    }
    free(isDominated);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void initializeCommonVariables()
{
    int i;

    initializeRandomNumberGenerator();
    generation_base                         = 2;

    number_of_generations                   = 0;
    number_of_evaluations                   = 0;
    objective_discretization_in_effect      = 0;
    elitist_archive_size                    = 0;
    elitist_archive_capacity                = 10;
    elitist_archive                         = (char **) Malloc( elitist_archive_capacity*sizeof( char * ) );
    elitist_archive_objective_values        = (double **) Malloc( elitist_archive_capacity*sizeof( double * ) );
    elitist_archive_constraint_values       = (double *) Malloc( elitist_archive_capacity*sizeof( double ) );
    elitist_archive_from                    = (int *) Malloc( elitist_archive_capacity*sizeof( int ) );
    elitist_archive_chromosome_index        = (int *) Malloc( elitist_archive_capacity*sizeof( int ) );
    elitist_archive_evolution_due           = (int *) Malloc( elitist_archive_capacity*sizeof( int ) );
    
    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        elitist_archive[i]                  = (char *) Malloc( number_of_parameters*sizeof( char ) );
        elitist_archive_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
    }
    elitist_archive_copy                    = NULL;
    objective_discretization = (double *) Malloc( number_of_objectives*sizeof( double ) );

    MI_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( i = 0; i < number_of_parameters; i++ )
        MI_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );

    population_indices_of_cluster_members   = NULL;
    population_cluster_sizes                = NULL;

    offspring = NULL;
    
    number_of_populations = 0;

    lt = NULL;
}

void ezilaitiniCommonVariables( void )
{
    int      i, j;
    
    if( elitist_archive_copy != NULL )
    {
        for( i = 0; i < elitist_archive_copy_size; i++ )
        {
            free( elitist_archive_copy[i] );
            free( elitist_archive_copy_objective_values[i] );
        }
        free( elitist_archive_copy );
        free( elitist_archive_copy_objective_values );
        free( elitist_archive_copy_constraint_values );
    }

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        free( elitist_archive[i] );
        free( elitist_archive_objective_values[i] );        
    }
    free( elitist_archive );
    free( elitist_archive_objective_values );
    free( elitist_archive_constraint_values );
    free( elitist_archive_from );
    free( objective_discretization );
    
    for( i = 0; i < number_of_parameters; i++ )
        free( MI_matrix[i] );

    free( MI_matrix );
}

void loadProblemData()
{
    switch(problem_index)
    {
        case ZEROMAX_ONEMAX: onemaxLoadProblemData(); break;
        case TRAP5: trap5LoadProblemData(); break;
        case LOTZ: lotzLoadProblemData(); break;
        case KNAPSACK: knapsackLoadProblemData(); break;
        case MAXCUT: maxcutLoadProblemData(); break;
        default: 
            printf("Cannot load problem data!\n");
            exit(1);
    }
}

void ezilaitiniProblemData()
{
    double **default_front;
    int i, default_front_size;

    switch(problem_index)
    {
        case KNAPSACK: ezilaitiniKnapsackProblemData(); break;
        case MAXCUT: ezilaitiniMaxcutProblemData(); break;
    }

    free(optimization);
    
    default_front = getDefaultFront( &default_front_size );
    if( default_front )
    {
        for( i = 0; i < default_front_size; i++ )
            free( default_front[i] );
        free( default_front );
    }
}

void run( void )
{
    loadProblemData();

    initializeCommonVariables();

    schedule();

    ezilaitiniProblemData();

    ezilaitiniCommonVariables();
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Main -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * The main function:
 * - interpret parameters on the command line
 * - run the algorithm with the interpreted parameters
 */
int main( int argc, char **argv )
{
    interpretCommandLine( argc, argv );

    run();

    return( 0 );
}
