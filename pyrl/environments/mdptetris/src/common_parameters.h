#ifndef COMMON_PARAMETERS_H
#define COMMON_PARAMETERS_H

#include "types.h"
#include "rewards.h"
#include "macros.h"
#include "file_tools.h"

#define MAX_LENGTH 256

/**
 * This structure defines the parameters used in the LPI and CE algorithms.
 */
struct CommonParameters {

  /* Parameters common to algorithms and tetris */

  int board_width;                               /* board width */
  int board_height;                              /* board height */

  int tetris_implementation;                     /* 0: Simplified Tetris, 1: RLC2008 Tetris, 2(to be done): Original Tetris */   

  int allow_lines_after_overflow;                /* enable the lines completion when the piece overflows? */
  char piece_file_name[MAX_FILE_NAME];           /* file defining the pieces */
  unsigned int random_generator_seed;            /* seed of the random number generator */

  /* Parameters which are common to LPI and CE algorithms */

  RewardDescription reward_description;          /* reward function */
};


void set_default_reward_function(RewardFunctionID reward_function_id);

void load_default_parameters(CommonParameters *parameters);

void ask_common_parameters(CommonParameters *parameters);

int parse_common_parameter(CommonParameters *parameters, int nb_args, char **args, void (*print_usage)(void));

void parameters_assert(int assertion, const char *error_message, void (*print_usage)(void));

void common_parameters_print_usage(void);

#endif
