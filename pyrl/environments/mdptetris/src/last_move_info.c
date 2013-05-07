#include "config.h"
#include "last_move_info.h"

/**
 * @brief Prints information about the last move.
 * @param out file to write
 * @param last_move_info some last move information to print
 */
void print_last_move_info(FILE *out, LastMoveInfo *last_move_info) {
  printf("Last move:\n");
  printf("  Removed lines: %d\n", last_move_info->removed_lines);
  printf("  Landing height: %d\n", last_move_info->landing_height_bottom);
  printf("  Eliminated cells from the last piece : %d\n", last_move_info->eliminated_bricks_in_last_piece);
}
