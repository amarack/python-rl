/**
 * @defgroup board Board
 * @ingroup api
 * @brief The Tetris board
 *
 * This module handles the game board. The board is composed by an array of rows.
 * Each row is represented by a 16-bit integer.
 *
 * @{
 */
#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include "types.h"
#include "piece.h"
#include "last_move_info.h"

/**
 * @brief The game board.
 *
 * The rows are numeroted from bottom to top, starting with 0.
 * The columns are numeroted from left to right, starting with 1 (actually column 0 is the left border).
 */
struct Board {
  /**
   * @name Board structure and content
   */
  int width;            /**< Number of columns in the board, not including the side borders (10 in standard Tetris). */
  int height;           /**< Number of rows in the board (20 in standard Tetris). */
  int extended_height;  /**< Number of rows in the internal representation of board (24 in standard Tetris). */
  int allow_lines_after_overflow;  /**< enable the lines completion when the piece overflows? */
  uint16_t *rows;       /**< Board state: array of rows where each row is represented with an integer. */

  /**
   * @name Information stored to improve the speed
   */
  int wall_height;      /**< Current height of the wall (index of the lowest empty row). */
  int max_piece_height; /**< Maximum height of a piece (4 for standard Tetris), used to know how many
                             lines we have to check when a piece is dropped. */
  int *column_heights;  /**< Height of each column. */

  /**
   * @name Bit masks depending on the board size
   */
  uint16_t empty_row;   /**< 16-bit integer representing an empty row (for the standard board size: 1000000000011111). */
  uint16_t full_row;    /**< 16-bit integer representing a full row (for the standard board size: 1111111111111111). */

  /**
   * @name Information needed to cancel the last move.
   */
  uint16_t *previous_rows;  /**< The board state before the last move. */
  int previous_wall_height; /**< The wall height (index of the first empty row) before the last move. */
};

/**
 * @name Board creation and destruction
 *
 * These functions allow to create or destroy a board.
 *
 * @{
 */
Board *new_board(int width, int height, int allow_lines_after_overflow, int nb_pieces, Piece *pieces);
Board *new_board_copy(const Board *board);
void free_board(Board *board);
/**
 * @}
 */


/**
 * @name Actions on the board
 *
 * These functions change the board state.
 * You should not change the content of a board's structure directly.
 *
 * @{
 */
int board_drop_piece(Board *board, PieceOrientation *oriented_piece, int orientation, int column, LastMoveInfo *last_move_info, int cancellable);
int board_drop_piece_fancy(Board *board, PieceOrientation *oriented_piece, int orientation, int column,LastMoveInfo *last_move_info, int cancellable, int **fancy_board);
int board_drop_piece_rlc(Board *board, Piece *pieces, int piece_index, int desired_orientation, int desired_column,LastMoveInfo *last_move_info, int cancellable);
void board_cancel_last_move(Board *board);
void board_reset(Board *board);
/**
 * @}
 */

/**
 * @name Additional information about the board
 *
 * The following functions provide some other information about the board.
 * For performance reasons, these information are computed only
 * if you ask it explicitly. 
 *
 * @{
 */
void board_update_column_heights(Board *board);
int board_get_column_height(Board *board, int column);
/**
 * @}
 */

/**
 * @name Displaying
 * @{
 */
void board_print(FILE *out, Board *board);
/**
 * @}
 */

#endif

/**
 * @}
 */
