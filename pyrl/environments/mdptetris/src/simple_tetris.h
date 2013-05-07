/**
 * This module defines the properties of a simplified
 * tetris game.
 */

#ifndef SIMPLE_TETRIS_H
#define SIMPLE_TETRIS_H

#include <stdint.h>
#include "game.h"
#include "macros.h"
#include "file_tools.h"

/* Uncomment the following line for 4*5  */
/*#define SIZE_45 */


#ifdef SIZE_45

#define WIDTH 4
#define HEIGHT 5
#define NB_STATES 1048576
#define LAST_BITS_MASK 0x0000000F /* 4 bits */
#define WEAK_BITS_SHIFT 11

#else

#define WIDTH 5
#define HEIGHT 5
#define NB_STATES 33554432
#define LAST_BITS_MASK 0x0000001F /* 5 bits */
#define WEAK_BITS_SHIFT 10

#endif

/**
 * Parameters of the algorithm.
 * This parameters are saved in the values file so the
 * algorithm can be resumed after an interruption.
 */
typedef struct ValueIterationParameters {
  int nb_pieces;      /* number of pieces */
  double gamma;       /* discount factor */
  double delta_limit; /* limit to stop the algorithm */
  int iterations;     /* current number of iterations */
  int use_buffer;     /* 1 to use a buffer */

  /* files */
  char piece_file_name[MAX_FILE_NAME];
  char delta_file_name[MAX_FILE_NAME];
} ValueIterationParameters;

typedef struct OldValueIterationParameters {
  int nb_pieces;      /* number of pieces */
  double gamma;       /* discount factor */
  double delta_limit; /* limit to stop the algorithm */
  int iterations;     /* current number of iterations */

  /* files */
  char piece_file_name[MAX_FILE_NAME];
  char delta_file_name[MAX_FILE_NAME];
} OldValueIterationParameters;

uint32_t get_game_code(Game *game);
void set_game_state(Game *game, uint32_t state_code);

#endif
