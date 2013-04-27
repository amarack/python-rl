/*   mdp.h  

  *****
  Copyright 1994-1997, Brown University
  Copyright 1998, 1999, Anthony R. Cassandra

                           All Rights Reserved
                           
  Permission to use, copy, modify, and distribute this software and its
  documentation for any purpose other than its incorporation into a
  commercial product is hereby granted without fee, provided that the
  above copyright notice appear in all copies and that both that
  copyright notice and this permission notice appear in supporting
  documentation.
  
  ANTHONY CASSANDRA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
  PARTICULAR PURPOSE.  IN NO EVENT SHALL ANTHONY CASSANDRA BE LIABLE FOR
  ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  *****

     header file for mdp.c
*/
#ifndef MDP_MDP_H
#define MDP_MDP_H

#include "sparse-matrix.h"

/* Use this type for a variable that indicated whether we have a 
   POMDP or an MDP.
*/
typedef enum {  UNKNOWN_problem_type, 
		MDP_problem_type, 
		POMDP_problem_type 
	      } Problem_Type;

/* Use this to determine if the problems values are rewards or costs.
   */
#define NUM_VALUE_TYPES          2
typedef enum {REWARD_value_type, COST_value_type } Value_Type;
#define VALUE_TYPE_STRINGS       { \
                                   "cost", \
                                   "reward" \
                                 }

#define DEFAULT_DISCOUNT_FACTOR               1.0

#define DEFAULT_VALUE_TYPE                    REWARD_value_type

#define INVALID_STATE                         -1
#define INVALID_OBS                           -1
#define INVALID_ACTION                        -1

#ifndef MDP_C

/* Exported variables */
extern char *value_type_str[];
extern double gDiscount;
extern Problem_Type gProblemType;
extern Value_Type gValueType;

/* We will use this flag to indicate whether the problem has negative
   rewards or not.  It starts off FALSE and becomes TRUE if any
   negative reward is found. */
extern double gMinimumImmediateReward;

extern int gNumStates;
extern int gNumActions;
extern int gNumObservations;

/* Intermediate variables */

extern I_Matrix *IP;  /* Transition Probabilities */
extern I_Matrix *IR;  /* Observation Probabilities */
extern I_Matrix IQ;  /* Immediate values for MDP only */

/* Sparse variables */

extern Matrix *P;  /* Transition Probabilities */
extern Matrix *R;  /* Observation Probabilities */
extern Matrix *QI;  /* The immediate values, for MDPs only */
extern Matrix Q;  /* Immediate values for state action pairs.  These
		     are expectations computed from immediate values:
		     either the QI for MDPs or the special
		     representation for the POMDPs */

extern double *gInitialBelief;   /* For POMDPs */
extern int gInitialState;        /* For MDPs   */

/* Exported functions */
extern double *newBeliefState();
extern int transformBeliefState( double *pi,
                                double *pi_hat,
                                int a,
                                int obs );
extern void copyBeliefState( double *copy, double *pi );
extern void displayBeliefState( FILE *file, double *pi );
extern int readMDP( char *filename );
extern void convertMatrices();
extern void deallocateMDP();
extern void convertMatrices();
extern int verifyIntermediateMDP();
extern void deallocateIntermediateMDP();
extern void allocateIntermediateMDP();
extern int writeMDP( char *filename );
extern void displayMDPSlice( int state );

#endif
#endif
