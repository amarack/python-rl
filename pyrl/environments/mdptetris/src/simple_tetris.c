#include "config.h"
#include "simple_tetris.h"
#include "brick_masks.h"

/**
 * Returns the state code corresponding to a game
 * without the current piece.
 */
uint32_t get_game_code(Game *game) {
  uint32_t state_code;
  uint16_t *board_rows;
  int i;

  state_code = 0;

  board_rows = game->board->rows;
  for (i = game->board->wall_height - 1; i >= 0; i--) {
    state_code = state_code << WIDTH;
    state_code |= (board_rows[i] & brick_masks_inv[0]) >> WEAK_BITS_SHIFT;
  }

  return state_code;
}

/**
 * Sets the game board (not including the current piece)
 * corresponding to a integer code.
 */
void set_game_state(Game *game, uint32_t state_code) {
  uint16_t *board_rows;
  uint16_t row, empty_row;
  int i;
  int wall_height;

  /* each bit represents the state of a cell */
  board_rows = game->board->rows;
  empty_row = game->board->empty_row;
  for (i = 0; i < HEIGHT; i++) {
    wall_height = i;
    row = (uint16_t) (state_code & LAST_BITS_MASK);
    if (!row) { /* the row is empty */
      board_rows[i] = empty_row;
    }
    else {
      board_rows[i] = (row << WEAK_BITS_SHIFT) | empty_row;
    }
    state_code = state_code >> WIDTH;
  }
  game->board->wall_height = i;
}
