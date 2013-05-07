#include "config.h"
#include "rewards.h"

RewardFunction *all_reward_functions[] = {
  get_no_reward,
  get_reward_removed_lines,
  get_reward_one,
  get_reward_at_least_one_line,
  get_reward_tetris_are_better
};

/**
 * Returns 0.
 */
int get_no_reward(Game *game) {
  return 0;
}

/**
 * Returns the number of lines removed in the last move.
 */
int get_reward_removed_lines(Game *game) {
  return game->last_move_info.removed_lines;
}

/**
 * Returns 1 if the game is not over.
 */
int get_reward_one(Game *game) {
  if (game->game_over) {
    return 0;
  } else {
    return 1;
  }
}

/**
 * Returns 1 if there was at least one removed line in the last move.
 */
int get_reward_at_least_one_line(Game *game) {
  return (game->last_move_info.removed_lines > 0) ? 1 : 0;  
}


/**
 * Returns a number which grows quickly with the number of removed lines. (useful for the RL competition)
 */
int get_reward_tetris_are_better(Game *game) {
  switch (game->last_move_info.removed_lines) {
  case 0:
    return 0;
  case 1:
    return 1;
  case 2:
    return 3;
  case 3:
    return 7;
  case 4:
    return 13;
  }
  return(0);
}

