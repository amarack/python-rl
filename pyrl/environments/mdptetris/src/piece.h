/**
 * @defgroup piece Pieces
 * @ingroup api
 * @brief Tetris pieces
 *
 * This module handles the game pieces. A piece is composed by an array of orientations.
 * Each orientation is an array of rows, where each row is represented by a 16-bit integer.
 *
 * @{
 */
#ifndef PIECE_H
#define PIECE_H

#include <stdio.h>
#include <stdint.h>
#include "types.h"

/**
 * @brief A Tetris piece oriented in a specific direction.
 */
struct PieceOrientation {
  int width;                  /**< Width of the piece in this orientation. */
  int height;                 /**< Height of the piece in this orientation. */
  uint16_t *bricks;           /**< Shape of the piece in this orientation
                               * (array of size \c height where each element is a
			       * 16-bit integer representing a row). */
  int *nb_full_cells_on_rows; /**< Number of full cells on each row (array of
			       * size \c height where each element is the
			       * number of full cells on a row. */
};

/**
 * @brief A Tetris piece with its possible orientations.
 */
struct Piece {
  int nb_orientations;            /**< Number of possible orientations of the shape. */
  PieceOrientation *orientations; /**< Array of size \c nb_orientations, containing
				   * the possible orientations of the piece
				   * (not an array of pointers) */
};

/**
 * @name Piece creation and destruction
 *
 * These functions allow to create or destroy the pieces.
 *
 * @{
 */
void load_pieces(const char *file_name, int *nb_pieces, Piece **pieces);
void free_piece(Piece *piece);
/**
 * @}
 */


/**
 * @name Displaying
 * 
 * These functions display human-readable views of a Tetris piece.
 */
void piece_print(FILE *out, Piece *piece);
void piece_print_orientation(FILE *out, PieceOrientation *orientation);
void piece_print_orientations(FILE *out, Piece *piece);
/**
 * @}
 */

#endif

/**
 * @}
 */
