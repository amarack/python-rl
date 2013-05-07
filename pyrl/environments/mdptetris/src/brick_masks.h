/**
 * Bit masks to store the board state and the shape of the pieces.
 */

#ifndef BRICK_MASKS_H
#define BRICK_MASKS_H

#include <stdio.h>
#include <stdint.h>

/**
 * Bit masks to represent the bricks on a row and the shape of each piece.
 * With this representation, a row state is stored on a single 16-bit integer.
 * A row size must not exceed 16 cells.
 * There is 12 cells in the row of a standard Tetris game (including the 2 side borders).
 * These bit fields are also used to represent the shape of the pieces.
 */
extern const uint16_t brick_masks[];
extern const uint16_t brick_masks_inv[];

void print_row(FILE *out, uint16_t row);

#endif
