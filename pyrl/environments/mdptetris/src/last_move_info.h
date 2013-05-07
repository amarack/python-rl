/**
 * @defgroup last_move_info Last move info
 * @ingroup api
 * @brief Information about the last move
 *
 * This module stores some information about the last move made.
 * This information is actually used by some features: indeed, several features
 * evaluate the last action instead of the current state itself.
 *
 * @{
 */
#ifndef LAST_MOVE_INFO_H
#define LAST_MOVE_INFO_H

#include <stdio.h>
#include "types.h"

/**
 * @brief Information about the last action.
 */
typedef struct LastMoveInfo {

  /**
   * @name Effect of the action on the board
   */
  int removed_lines;                      /**< Number of rows completed during the move
					   * (also used to cancel a move). */
  int landing_height_bottom;              /**< Index of the row where the bottom part
					   * of the piece is put. */
  int eliminated_bricks_in_last_piece;    /**< Number of cells of the last piece put
					   * that were part of rows completed. */

  /**
   * @name The action made
   */
  int column;                             /**< Column where the last piece was put
					   * (\c 1 to \c w where \c w is the board size). */
  int orientation;                        /**< Orientation of the last piece (\c 0 to
					   * <code>n - 1</code> where \c n is the number of
					   * possible orientations of the piece. */
  PieceOrientation *oriented_piece;       /* The last piece put. */

  int nb_steps;                           /* The number of steps (RLC mode) */

} LastMoveInfo;

/**
 * @name Displaying
 * @{
 */
void print_last_move_info(FILE *out, LastMoveInfo *last_move_info);
/**
 * @}
 */

#endif
/**
 * @}
 */
