#include <stdlib.h>
#include "config.h"
#include "feature_functions.h"
#include "feature_policy.h"
#include "game.h"
#include "simple_tetris.h"
#include "brick_masks.h"
#include "macros.h"

/**
 * @brief Number of possible states on a single row.
 *
 * A row is represented by a 16-bit integer so there are
 * 2^16 possible rows, even if many of them will never occur
 * because some bits are always 1.
 */
#define NB_POSSIBLE_ROWS 65536

/**
 * @brief All feature functions.
 *
 * Associates a feature function to each feature index.
 */
static FeatureFunction * const all_features[] = {

  get_diversity,                                      /* -14 */

  get_next_column_height2,                            /* -13  */
  
  get_height_square,                                  /* -12  */
  get_hole_depths_square,                             /* -11  */
  get_wall_distance_to_top_square,                    /* -10  */ 

  get_next_column_height_difference2,          /* -9  */
  get_next_column_distance_to_top,             /* -8  */
  get_wall_distance_to_top,                    /* -7  */ 
    
  get_well_sums_fast,                 /* -6 */
  get_next_local_value_function,      /* -5 */
  get_surrounded_holes,               /* -4 */
  get_next_wall_height,               /* -3 */
  get_rows_with_holes,                /* -2 */
  get_hole_depths,                    /* -1 */

  get_constant,                       /* 0  */

  get_landing_height,                 /* 1  */
  get_eroded_piece_cells,             /* 2  */
  get_row_transitions,                /* 3  */
  get_column_transitions,             /* 4  */
  get_holes,                          /* 5  */
  get_well_sums_dellacherie,          /* 6  */

  get_wall_height,                    /* 7  */
  get_next_column_height,             /* 8  */
  get_next_column_height_difference,  /* 9  */

  get_occupied_cells,                 /* 10 */
  get_weighted_cells,                 /* 11 */
  get_wells,                          /* 12 */
  get_rows_eliminated,                /* 13 */

  NULL,                               /* 14 (not yet implemented) */
  get_max_height_difference,          /* 15 */

  

  /* the rest is not yet implemented */

};

/**
 * @brief True if the feature system has been initialized
 * (i.e. features_initialized has been called).
 */
static int initialized = 0;

/**
 * @brief Stores for each possible row its number of transitions.
 */
static char row_transitions[NB_POSSIBLE_ROWS];

/**
 * @brief Stores for each possible 16-bit number its number of bits 1.
 */
static char bits_1[NB_POSSIBLE_ROWS];

/**
 * @brief The matrix of state values for feature NEXT_LOCAL_VALUE_FUNCTION.
 */
static double *local_value_function = NULL;

static void initialize_next_local_value_function();

/**
 * @brief Returns a feature function.
 *
 * This function associates a feature function to each feature index.
 *
 * @param feature_id id of the feature function to get
 * @return the corresponding feature function
 */
FeatureFunction *feature_function(FeatureID feature_id) {

  return all_features[feature_id + NB_ORIGINAL_FEATURES];
}

/**
 * @brief Initializes the feature functions system.
 *
 * This function must be called before any feature function is used,
 * otherwise many of them won't work
 * (get_holes(), get_row_transitions(), get_column_transitions()...).
 *
 * @param feature_policy the features, needed here because the initializations made
 * depend on what features are present
 * @see features_exit()
 */
void features_initialize(const FeaturePolicy *feature_policy) {
  uint16_t row, current_bit, previous_bit;
  int transitions, nb_bits_1, i, j;

  if (!initialized) {

    /* count row_transitions and bits_1 in the same loop */
    for (i = 0; i < NB_POSSIBLE_ROWS; i++) {

      transitions = 0;
      nb_bits_1 = 0;
      row = i;
      previous_bit = row & brick_masks[0];
      if (previous_bit) {
	nb_bits_1++;
      }
      for (j = 1; j < 16; j++) {
	row = row << 1;
	current_bit = row & brick_masks[0];

	if (current_bit ^ previous_bit) {
	  transitions++;
	}

	if (current_bit) {
	  nb_bits_1++;
	}

	previous_bit = current_bit;
      }
      row_transitions[i] = transitions;
      bits_1[i] = nb_bits_1;
      /*     print_row(stdout, i); */
      /*     printf(": nb transitions: %d, nb bits 1: %d\n\n", transitions, nb_bits_1); */
    }
  }

  /* if feature NEXT_LOCAL_VALUE_FUNCTION is present, we have to load the value function file */
  if (local_value_function == NULL
      && contains_feature(feature_policy, NEXT_LOCAL_VALUE_FUNCTION)) {
    
    initialize_next_local_value_function();
  }

  initialized = 1;
}

/**
 * @brief Initializes feature NEXT_LOCAL_VALUE_FUNCTION.
 *
 * This function is called if feature NEXT_LOCAL_VALUE_FUNCTION is present.
 * The value function of the 5*5 board is loaded from
 * the file \c "results_value_iteration/5_5.values".
 */
static void initialize_next_local_value_function() {

  FILE *value_file;
  const char *file_name = "results_value_iteration/5_5.values";

  value_file = fopen(file_name, "r");
  
  if (value_file == NULL) {
    DIE1("Unable to open the value file '%s'\n", file_name);
  }

  printf("Loading the value function of the 5*5 board...\n");

  MALLOCN(local_value_function, double, NB_STATES);
  fseek(value_file, sizeof(ValueIterationParameters), SEEK_SET);
  FREAD(local_value_function, sizeof(double), NB_STATES, value_file);
  fclose(value_file);

  printf("Value function loaded\n");
}

/**
 * @brief Frees the memory used by the feature functions system.
 * @see features_initialize()
 */
void features_exit(void) {

  if (local_value_function != NULL) {
    FREE(local_value_function);
  }
  initialized = 0;
}

/**
 * @brief Feature #0: Returns always 1.
 *
 * This feature function is used by reinforcement learning approaches.
 *
 * @param game the current game state
 * @return 1
 */
double get_constant(Game *game) {
  return 1;
}

/**
 * @brief Feature #1 (Dellacherie): Returns the height where the last piece was put.
 *
 * The middle of the piece is considered.
 *
 * @param game the current game state
 * @return the height where the last piece was put.
 */
double get_landing_height(Game *game) {
  
  if (game->last_move_info.oriented_piece != NULL) { /* if not the first move */

    return game->last_move_info.landing_height_bottom + ((game->last_move_info.oriented_piece->height - 1) / 2.0);
  }
  else {
    return 0;
  }
}

/**
 * @brief Feature #2 (Dellacherie): Evaluates how the last piece contributed to remove lines.
 *
 * The value returned is <code>n * m</code>, where \c n is the number of lines
 * removed during the last move, and \c m is the number
 * of cells of the last piece that were eliminated in the move.
 *
 * @param game the current game state
 * @return a value indicating how the last piece contributed to remove lines.
 */
double get_eroded_piece_cells(Game *game) {
  return game->last_move_info.removed_lines * game->last_move_info.eliminated_bricks_in_last_piece;
}

/**
 * @brief Feature #3 (Dellacherie): Returns the number of irregularities in the rows.
 *
 * A row transition occurs when a full cell is adjacent to an empty one on the same row.
 * The two side walls are considered as full cells: thus, an empty row has
 * two transitions.
 *
 * @param game the current game state
 * @return the number of row transitions
 * @see get_column_transitions()
 */
double get_row_transitions(Game *game) {
  int i, wall_height, board_height, result;
  uint16_t *board_rows;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  board_height = board->height;
  wall_height = board->wall_height;
  result = 0;
  for (i = 0; i < wall_height; i++) {
    result += row_transitions[board_rows[i]];
  }
  /* count the remaining rows */
  result += 2 * (board_height - wall_height);

  return result;
}

/**
 * @brief Feature #4 (Dellacherie): Returns the number of irregularilities in the columns.
 *
 * This function is just like get_row_transitions(), but for the columns.
 * A column transition occurs when a full cell is adjacent to an empty one on the same column.
 * The floor (the bottom row) is considered as full cells: thus, an empty column has
 * one transition.
 *
 * @param game the current game state
 * @return the number of column transitions
 * @see get_row_transitions()
 */
double get_column_transitions(Game *game) {
  int i, wall_height, board_height, transitions;
  uint16_t *board_rows, current_row, previous_row, xor;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  board_height = board->height;
  wall_height = board->wall_height;

  previous_row = board->full_row;
  transitions = 0;
  for (i = 0; i < wall_height; i++) {
    current_row = board_rows[i];
 
    /* make a XOR between the current row and the previous row */
    xor = current_row ^ previous_row;

    /* count the number of 1 in the bits of xor */
    transitions += bits_1[xor];

    previous_row = current_row;
  }
  /* don't forget the last row */
  current_row = board->empty_row;
  xor = current_row ^ previous_row;
  transitions += bits_1[xor];

  return transitions;
}

/**
 * @brief Feature #5: Returns the number of holes in the board.
 *
 * A hole is an empty cell such that there exists at least one full cell
 * in the same column above it.
 *
 * @param game the current game state
 * @return the number of holes in the board
 */
double get_holes(Game *game) {
  int i, holes, wall_height;
  uint16_t *board_rows, row_holes, previous_row, current_row;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  wall_height = board->wall_height;

  if (wall_height <= 1) {
    return 0;
  }

  previous_row = board_rows[wall_height - 1];
  holes = 0;
  row_holes = 0x0000; /* the bits 1 indicate the holes on the current row */

  for (i = wall_height - 2; i >= 0; i--) {
    current_row = board_rows[i];

    /**
     * A cell is a hole if it is empty and the cell above is full or already a hole.
     * We compute that with bit-to-bit operations to consider all columns simultaneously.
     */
    row_holes = ~current_row & (previous_row | row_holes);

    /**
     * The bits 1 in row_holes indicates the holes.
     * We just count their number to know the number of holes on the current row.
     */
    holes += bits_1[row_holes];

    previous_row = current_row;
  }

  return holes;
}

/**
 * @brief Feature #6 (Dellacherie): Evaluates the wells and how deep they are.
 *
 * A well is a place which can be filled only by vertical bars.
 * This is the exact original feature as proposed by Dellacherie.
 *
 * We call "well cell" an empty cell such that
 * its left cell and its right cell are both occupied.
 * This feature is computed as follows.
 * For each column, we search all well cells.
 * For each well cell, we count the number of successive empty cells
 * below it, including the well cell itself.
 *
 * Example:
 *
 * |........|
 * |........|
 * |...X....|
 * |...XX...|
 * |.X.XX...|
 * |.XXXXX.X|
 * |.XXXXX.X|
 * |XXXXX..X|
 * |...XXX.X|
 * |X.X..X.X|
 *
 * In this board, we have 9 well cells. Let's represent them by an 'o':
 *
 * |........|
 * |........|
 * |...X....|
 * |...XX...|
 * |oXoXX...|
 * |oXXXXXoX|
 * |oXXXXXoX|
 * |XXXXX..X|
 * |...XXXoX|
 * |XoX..XoX|
 *
 * We detail the execution of the algorithm on this example:
 *
 * - Column 1: value = 6
 *
 * In column 1, we have 3 well cells. The topmost one has a value of 3
 * (because there are 2 unoccupied cells below it, plus itself).
 * Similarly, the middle well cell if this column has a value of 2,
 * and the last one has a value of 1.
 * Then the total of column is 3 + 2 + 1 = 6.
 *
 * - Column 2: value = 1
 *
 * In column 2, the only well cell has no unoccupied cell below it, so its value is 1.
 *
 * - Column 3: value = 1
 *
 * Same as in column 2.
 *
 * - Column 7: value = 12
 *
 * The topmost well cell has a value of 5 because there are 4 empty cells below it.
 * The three other values are 4, 2 and 1. Note that there are only 4 well cells in this column.
 *
 * 5 + 4 + 2 + 1 = 12
 *
 * - Total: 6 + 1 + 1 + 12 = 20
 *
 * @param game the current game state
 * @return a value indicating how deep are the wells if any
 */
double get_well_sums_dellacherie(Game *game) {
  int i, i2, j, board_width, wall_height, result;
  uint16_t *board_rows;
  uint16_t well_mask;
  uint16_t well_pattern;
  Board *board;

  board = game->board;
  board_width = board->width;
  board_rows = board->rows;
  wall_height = board->wall_height;

  result = 0;
  well_mask = 0xE000; /* 1110000000000000 */
  well_pattern = 0xA000; /* 1010000000000000 */
  for (j = 1; j <= board_width; j++) {

    /* ******************* FIX ME **********************: should be for (i = wall_height ; i >= 0; i--) { */
    for (i = wall_height - 1; i >= 0; i--) {

      if ((board_rows[i] & well_mask) == well_pattern) { /* there is a well */
	
	result++;
	i2 = i-1;

	while (i2 >= 0 && !(board_rows[i2] & brick_masks[j])) { /* stops when the bottom of the well is reached */
	  
	  result++;
	  i2--;	  

	}
      }
    }
    well_mask = well_mask >> 1;
    well_pattern = well_pattern >> 1;
  }
  return result;
}

/**
 * @brief Feature #7 (Bertsekas): Returns the wall height.
 *
 * The wall height is index of the lowest empty row.
 * The wall height of an empty board is zero.
 *
 * @param game the current game state
 * @return the wall height
 */
double get_wall_height(Game *game) {
  return game->board->wall_height;
}

/**
 * @brief Feature #8 (Bertsekas): Returns the height of a column.
 *
 * This feature function should be called \c w times, where
 * \c w is the board width. When this function is called again,
 * the next column is considered.
 *
 * Warning: this function calls board_get_column_height(), so
 * board_update_column_heights() must have been called first
 * (if you use a \ref feature_policy "feature policy",
 * this is already done for you).
 *
 * @param game the current game state
 * @return the height of the current column
 */
double get_next_column_height(Game *game) {
  static int current_column = 1;
  int result;

/*   game_print(stdout, game); */
/*   printf("current column = %d, height = %d\n", current_column, game->board->column_heights[current_column]); */
/*   getchar(); */

  result = game->board->column_heights[current_column];
  current_column = (current_column % game->board->width) + 1;

  return result;
}

/**
 * @brief Feature #9 (Bertsekas): Returns the difference of height between
 * a column and the next one.
 *
 * This feature function should be called <code>w - 1</code> times, where
 * \c w is the board width. When this function is called again, the next two
 * columns are considered.
 *
 * Warning: this function calls board_get_column_height(), so
 * board_update_column_heights() must have been called first
 * (if you use a \ref feature_policy "feature policy",
 * this is already done for you).
 *
 * @param game the current game state
 * @return the difference of height between the current column and the next one
 */
double get_next_column_height_difference(Game *game) {
  static int current_column = 1;
  int result;
  int *column_heights;

  column_heights = game->board->column_heights;

  result = abs(column_heights[current_column] - column_heights[current_column + 1]);
  current_column = (current_column % (game->board->width - 1)) + 1;
  
  return result;
}

/**
 * @brief Feature #-3 (original): Returns 1 if the wall height is n,
 * and 0 otherwise, where n is the number
 * of times you already called this function in the current state.
 *
 * The first time, it returns \c 1 if the wall height is \c 0, and \c 0 otherwise.
 * When this function is called again, it returns \c 1 if the wall height is \c 1,
 * and 0 otherwise. So on and so forth: on the \c nth call, it returns \c 1 if the
 * wall height is <code>n-1</code>, and 0 otherwise.
 * So this function should be called \c h times, where \c h is the board height.
 *
 * This feature function groups together the states having the same wall height,
 * so we can assign a weight for each wall height. However, we didn't obtain good
 * results so far with this approach.
 *
 * @param game the current game state
 * @return \c 1 if the wall height is \n, and 0 otherwise, where \n is the number
 * of times you already called this function in the current state
 */
double get_next_wall_height(Game *game) {
  static int current_value = 0;
  int result;
  
  /*  printf("%d %d\n",game->board->wall_height,current_value); */

  if (game->board->wall_height == current_value) {
    result=1;
  } else {
    result=0;
  }
  current_value = (current_value+1) % (game->board->height+1);
  
  return result;
}

/**
 * @brief Feature #-1 (original): Returns the sum of hole depths.
 *
 * The depth of a hole is the number of successive full cells right above the hole.
 *
 * @param game the current game state
 * @return the sum of hole depths.
 */
double get_hole_depths(Game *game) {
  Board *board;
  int wall_height, result, i;
  uint16_t *board_rows, current_row, previous_row;
  uint16_t above_holes; /* each cell above a hole on the current row has bit 1 */

  board = game->board;
  wall_height = board->wall_height;
  board_rows = board->rows;

  result = 0;

/*   printf("\n------------------------\n"); */
/*   print_board(stdout, game->board); */

  above_holes = 0x0000;
  previous_row = board_rows[0];
  for (i = 1; i < wall_height; i++) {
    current_row = board_rows[i];

    /* detect the full cells above a hole */
    above_holes = current_row & (~previous_row | above_holes);

/*     printf("\nrow %d: above_holes = ", i); */
/*     print_row(stdout, above_holes); */
/*     printf("\n"); */
/*     getchar(); */

    /* count their number (i.e. the bits 1 in above_holes) */
    result += bits_1[above_holes];

    previous_row = current_row;
  }

  return result;
}

/**
 * @brief Feature #-4 (original): Returns the number of empty cells
 * having full cells both above and below.
 *
 * This feature function does not seem very relevant.
 *
 * @param game the current game state
 * @return the number of surrounded holes
 */
double get_surrounded_holes(Game *game) {
  int i, holes, wall_height;
  uint16_t *board_rows, row_holes, previous_row1, previous_row2, current_row;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  wall_height = board->wall_height;

  if (wall_height <= 1) {
    return 0;
  }

  previous_row2 = board_rows[wall_height - 1];
  previous_row1 = board_rows[wall_height - 2];
  holes = 0;
  row_holes = 0x0000; /* the bits 1 indicates the holes on previous_row1 */

  for (i = wall_height - 3; i >= 0; i--) {
    current_row = board_rows[i];

    /**
     * A cell is a hole if it is empty, and if the cells above and below are full.
     * We compute that with bit-to-bit operations to consider all columns simultaneously.
     */
    row_holes = current_row & ~previous_row1 & previous_row2;

    /**
     * The bits 1 in row_holes indicates the holes.
     * We just count their number to know the number of holes on the current row.
     */
    holes += bits_1[row_holes];

    previous_row2 = previous_row1;
    previous_row1 = current_row;
  }

  /* consider the row -1 as full*/
  row_holes = board->full_row & ~previous_row1 & previous_row2;
  holes += bits_1[row_holes];
  
  /* debug */
/*   game_print(stdout, game); */
/*   printf("Surrounded holes: %d\n", holes); */
/*   getchar(); */

  return holes;
}

/**
 * @brief Feature #-2 (original): Returns the number of rows having at least one hole.
 * @param game the current game state
 * @return the number of rows having at least one hole
 */
double get_rows_with_holes(Game *game) {

  int i, rows_with_holes, wall_height;
  uint16_t *board_rows, row_holes, previous_row, current_row;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  wall_height = board->wall_height;

  if (wall_height <= 1) {
    return 0;
  }

  previous_row = board_rows[wall_height - 1];
  rows_with_holes = 0;
  row_holes = 0x0000; /* the bits 1 indicate the holes on the current row */

  for (i = wall_height - 2; i >= 0; i--) {
    current_row = board_rows[i];

    /**
     * A cell is a hole if it is empty and the cell above is full or already a hole.
     * We compute that with bit-to-bit operations to consider all columns simultaneously.
     */
    row_holes = ~current_row & (previous_row | row_holes);

    /**
     * The bits 1 in row_holes indicates the holes.
     * We just count the number of rows with at least one hole.
     */
    if (row_holes != 0x0000) {
      rows_with_holes ++;
    }

    previous_row = current_row;
  }

  return rows_with_holes;
}

/**
 * @brief Feature #-5 (original): Returns the exact result of the value
 * function applied to a 5*5 subboard.
 *
 * This feature function should be called \c n times, where \c n is the
 * number of possible positions of a 5*5 window on the board (96 with
 * the standard board).
 * When this function is called again, the window is moved to the next
 * possible position.
 *
 * @param game the current game state
 * @return the value function of the current 5*5 subboard
 */
double get_next_local_value_function(Game *game) {

  /* position of the 5*5 window on the board */
  static int local_window_x = 0;
  static int local_window_y = 0;

  uint16_t *board_rows;
  uint16_t local_state_code;
  int i;

  /* first execution */
  if (local_window_y == 0) {
    local_window_y = game->board->height;
  }

  /* compute the integer number representing the local state */
  local_state_code = 0;

  board_rows = game->board->rows;
  for (i = local_window_y - 1; i >= local_window_y - HEIGHT; i--) {
    local_state_code = local_state_code << WIDTH;
    local_state_code |= (board_rows[i] >> (15 - local_window_x - WIDTH)) & LAST_BITS_MASK;

/*     print_row(stdout, board_rows[i]); */
/*     printf(" with x = %d -> ", local_window_x); */
/*     print_row(stdout, (board_rows[i] >> (15 - local_window_x - WIDTH)) & LAST_BITS_MASK); */
/*     printf(" or %x\n", (board_rows[i] >> (15 - local_window_x - WIDTH)) & LAST_BITS_MASK); */
  }
  
/*   print_board(stdout, game->board); */
/*   printf("x = %d, y = %d, code = %x, value = %f\n", local_window_x, local_window_y, local_state_code, local_value_function[local_state_code]); */
/*   printf("\n"); */
  
  /* update the window position */
  local_window_x++;

  if (local_window_x > game->board->width - WIDTH) {
    local_window_x = 0;
    local_window_y--;

    if (local_window_y < HEIGHT) {
      local_window_y = game->board->height;
    }
  }

  return local_value_function[local_state_code];
}
/**
 * @brief Feature #-6 (original): Evaluates the wells and how deep they are.
 *
 * This feature is almost the same as feature #6: WELL_SUMS_DELLACHERIE.
 * The only difference is that we consider that all empty cells below a well cell,
 * are well cells too. The algorithm is simpler because we can compute sums of
 * the form 1 + 2 + ... + n instead of considering the well cells one by one.
 * This version is slightly different from the original feature by Dellacherie
 * and does not return the same value for some boards.
 *
 * Example:
 *
 * |........|
 * |........|
 * |...X....|
 * |...XX...|
 * |.X.XX...|
 * |.XXXXX.X|
 * |.XXXXX.X|
 * |XXXXX..X|
 * |...XXX.X|
 * |X.X..X.X|
 *
 * In this board, we have 10 well cells (only 9 in Dellacherie's feature).
 * Let's represent them by an 'o':
 *
 * |........|
 * |........|
 * |...X....|
 * |...XX...|
 * |oXoXX...|
 * |oXXXXXoX|
 * |oXXXXXoX|
 * |XXXXX.oX|
 * |...XXXoX|
 * |XoX..XoX|
 *
 * There is a difference in column 7. Its value is 5 + 4 + 3 + 2 + 1 = 15
 * instead of 5 + 4 + 2 + 1 = 12.
 *
 * @param game the current game state
 * @return a value indicating how deep are the wells if any
 */
double get_well_sums_fast(Game *game) {
  int i, j, k, board_width, wall_height, result;
  uint16_t *board_rows;
  uint16_t well_mask;
  uint16_t well_pattern;
  Board *board;

  board = game->board;
  board_width = board->width;
  board_rows = board->rows;
  wall_height = board->wall_height;

  result = 0;
  well_mask = 0xE000; /* 1110000000000000 */
  well_pattern = 0xA000; /* 1010000000000000 */
  for (j = 1; j <= board_width; j++) {

    for (i = wall_height - 1; i >= 0; i--) {

      if ( (board_rows[i] & well_mask) == well_pattern ) { /* there is a well */
	
	result += 1;
	i--;
	k = 1;
	while (i >= 0 && !(board_rows[i] & brick_masks[j])) {
	  i--;
	  k++;
	  result += k;
	}
      }
    }
    well_mask = well_mask >> 1;
    well_pattern = well_pattern >> 1;
  }
  return result;
}

/**
 * @brief Feature #10 (Fahey): Returns the total number of occupied cells on the board.
 * @param game the current game state
 * @return the number of occupied cells
 */
double get_occupied_cells(Game *game) {

  int i, wall_height, occupied_cells;
  uint16_t *board_rows;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  wall_height = board->wall_height;

  occupied_cells = 0;

  for (i = 0; i < wall_height; i++) {
    occupied_cells += bits_1[board_rows[i]] + board->width - 16;
  }

  return occupied_cells;
}

/**
 * @brief Feature #11 (Fahey): Returns the total number of occupied cells on the board,
 * weighted by their height.
 * @param game the current game state
 * @return the number of occupied cells
 */
double get_weighted_cells(Game *game) {

  int i, wall_height, weighted_cells;
  uint16_t *board_rows;
  Board *board;

  board = game->board;
  board_rows = board->rows;
  wall_height = board->wall_height;

  weighted_cells = 0;

  for (i = 0; i < wall_height; i++) {
    weighted_cells += (bits_1[board_rows[i]] + board->width - 16) * (i + 1);
  }

  return weighted_cells;
}

/**
 * @brief Feature #12 (Fahey): Return the number of well cells.
 *
 * Here the well cells are defined as in feature #-6 (WELL_SUMS_FAST).
 *
 * @param game the current game state
 * @return the sum of well depths
 */
double get_wells(Game *game) {
  int i, j, board_width, wall_height, result;
  uint16_t *board_rows;
  uint16_t well_mask;
  uint16_t well_pattern;
  Board *board;

  board = game->board;
  board_width = board->width;
  board_rows = board->rows;
  wall_height = board->wall_height;

  result = 0;
  well_mask = 0xE000; /* 1110000000000000 */
  well_pattern = 0xA000; /* 1010000000000000 */
  for (j = 1; j <= board_width; j++) {

    for (i = wall_height - 1; i >= 0; i--) {

      if ((board_rows[i] & well_mask) == well_pattern) { /* there is a well */
	
	result++;
	i--;
	while (i >= 0 && !(board_rows[i] & brick_masks[j])) {
	  result++;
	  i--;
	}
      }
    }
    well_mask = well_mask >> 1;
    well_pattern = well_pattern >> 1;
  }
  return result;
}

/**
 * @brief Feature #13 (Fahey): Return the number of rows removed in the last move.
 *
 * @param game the current game state
 * @return the number of rows removed in the last move
 */
double get_rows_eliminated(Game *game) {
  return game->last_move_info.removed_lines;
}

/**
 * @brief Feature #15 (Bohm): Return the maximum height difference between two columns.
 *
 * @param game the current game state
 * @return the maximum height difference between two columns
 */
double get_max_height_difference(Game *game) {

  int height, previous_height, i;
  int min, max;
  Board *board = game->board;
  int width = board->width;

  max = -1;
  min = board->extended_height + 1;
  previous_height = board->column_heights[1];
  for (i = 1; i <= width; i++ ) {

    height = board->column_heights[i];
    
    if (height > max) {
      max = height;
    }

    if (height < min) {
      min = height;
    }
  }

  /* debug
  game_print(stdout, game);
  printf("max height diff = %d\n\n", max - min);
  */

  return max - min;
}



/**
 * @brief Feature #-7: Returns the wall distance to top.
 *
 * The wall distance_to_top is index of the lowest empty row.
 * The wall distance_to_top of an empty board is zero.
 *
 * @param game the current game state
 * @return the wall distance_to_top
 */
double get_wall_distance_to_top(Game *game) {
  return (game->board->height - game->board->wall_height);
}


/**
 * @brief Feature #-8 (Bertsekas): Returns the distance_to_top of the columns 1,2,average,n-1,n-2
 *
 * This feature function should be called \c w times, where
 * \c w is the board width. When this function is called again,
 * the next column is considered.
 *
 * Warning: this function calls board_get_column_height(), so
 * board_update_column_heights() must have been called first
 * (if you use a \ref feature_policy "feature policy",
 * this is already done for you).
 *
 * @param game the current game state
 * @return the height of the current column
 */
double get_next_column_distance_to_top(Game *game) {
  static int current_column = 1;
  double result;

  /* game_print(stdout, game);  */
/*   printf("current column = %d, height = %d\n", current_column, game->board->height - game->board->column_heights[current_column]);  */
/*   getchar();  */
  /* printf("%i %i  ",current_column,game->board->height - game->board->column_heights[current_column]);*/

  if ((current_column<=2) || (current_column>=game->board->width-1)) {
    
    result = game->board->height - game->board->column_heights[current_column];
    /*result = game->board->column_heights[current_column];*/
    current_column = (current_column % game->board->width) + 1;
  
  } else { /* current_column=3 */
  
    result=0.0;
    for (current_column = 3; current_column < game->board->width-1; current_column++) {
      result += game->board->height - game->board->column_heights[current_column];
      /*      result += game->board->column_heights[current_column];*/
    }
    result /= game->board->width-4;

  }

  
  return result;
}

/**
 * @brief Feature #-9: Returns the difference of height between
 * a column and the next one, for the column 1,2,average of 3-n-3,n-2,n-1
 *
 * This feature function should be called <code>w - 1</code> times, where
 * \c w is the board width. When this function is called again, the next two
 * columns are considered.
 *
 * Warning: this function calls board_get_column_height(), so
 * board_update_column_heights() must have been called first
 * (if you use a \ref feature_policy "feature policy",
 * this is already done for you).
 *
 * @param game the current game state
 * @return the difference of height between the current column and the next one
 */
double get_next_column_height_difference2(Game *game) {
  static int current_column = 1;
  double result;
  int *column_heights;

  column_heights = game->board->column_heights;

  if ((current_column<=2) || (current_column>=game->board->width-2)) {
    
    result = abs(column_heights[current_column] - column_heights[current_column + 1]);
    current_column = (current_column % (game->board->width - 1)) + 1;

  } else { /* current_column=3 */

    result=0.0;
    for (current_column = 3; current_column < game->board->width-2; current_column++) {  
      result += abs(column_heights[current_column] - column_heights[current_column + 1]);
    }
    result /= game->board->width-5;    

  }  
  
  return result;
}


/**
 * @brief Feature #-10:
 *
 */
double get_wall_distance_to_top_square(Game *game) {

  double i=get_wall_distance_to_top(game);
  return (i*i);

}

/**
 * @brief Feature #-11:
 *
 */
double get_height_square(Game *game) {

  double i=get_wall_height(game);
  return (i*i);

}

/**
 * @brief Feature #-12:
 *
 */
double get_hole_depths_square(Game *game) {

  double i=get_hole_depths(game);
  return (i*i);

}


/**
 * @brief Feature #-13 (Bertsekas): Returns the heights of columns 1,2,average(3->n-2),n-1,n-2
 *
 * This feature function should be called \c w times, where
 * \c w is the board width. When this function is called again,
 * the next column is considered.
 *
 * Warning: this function calls board_get_column_height(), so
 * board_update_column_heights() must have been called first
 * (if you use a \ref feature_policy "feature policy",
 * this is already done for you).
 *
 * @param game the current game state
 * @return the height of the current column
 */
double get_next_column_height2(Game *game) {
  static int current_column = 1;
  double result;

  /* game_print(stdout, game);  */
/*   printf("current column = %d, height = %d\n", current_column, game->board->height - game->board->column_heights[current_column]);  */
/*   getchar();  */
  /* printf("%i %i  ",current_column,game->board->height - game->board->column_heights[current_column]);*/

  if ((current_column<=2) || (current_column>=game->board->width-1)) {
    
    result = game->board->column_heights[current_column];
    /*result = game->board->column_heights[current_column];*/
    current_column = (current_column % game->board->width) + 1;
  
  } else { /* current_column=3 */
  
    result=0.0;
    for (current_column = 3; current_column < game->board->width-1; current_column++) {
      result += game->board->column_heights[current_column];
      /*      result += game->board->column_heights[current_column];*/
    }
    result /= game->board->width-4;

  }

  
  return result;
}


/**
 * @brief Feature #-14 (Bertsekas): Returns a coefficient between 0 and 5 that represents the diversity of the top of the wall: counts 1 if the following wall difference are seen: -2,-1,0,1,2
 *
 * Warning: this function calls board_get_column_height(), so
 * board_update_column_heights() must have been called first
 * (if you use a \ref feature_policy "feature policy",
 * this is already done for you).
 *
 * @param game the current game state
 * @return a coefficient between 0 and 5 that represents the diversity of the top of the wall
 */
double get_diversity(Game *game) {

  int i,width;
  int d_2=0,d_1=0,d0=0,d1=0,d2=0;
  int *column_heights;
  Board *board;
  
  board=game->board;
  column_heights = board->column_heights;
  width = board->width;

  for (i = 1; i < width; i++ ) {

    switch(column_heights[i+1]-column_heights[i]) {

    case -2:
      d_2=1;
      break;

    case -1:
      d_1=1;
      break;

    case 0:
      d0=1;
      break;

    case 1:
      d1=1;
      break;

    case 2:
      d2=1;
      break;

    }

  }
  
  return(d_2+d_1+d0+d1+d2);

}
