/**
 * This module provides some usual reward functions.
 */

#ifndef REWARD_H
#define REWARD_H

#include "types.h"
#include "game.h"

/**
 * Function type for a reward function.
 */
typedef int (RewardFunction)(Game *game);

/**
 * Constants to identify the reward functions.
 */
typedef enum {
  NO_REWARD,                /* always zero */
  REWARD_REMOVED_LINES,     /* number of lines removed in the last move */
  REWARD_ONE,               /* always 1 (rewards the number of moves before the game is over) */
  REWARD_AT_LEAST_ONE_LINE,  /* 1 if there was at least one line removed in the last move */
  REWARD_TETRIS_ARE_BETTER  /* 0,1,4,9,15 (for RL competition) */
} RewardFunctionID;

/**
 * This structure defines a reward function with its ID.
 */
struct RewardDescription {
  RewardFunction *reward_function;
  RewardFunctionID reward_function_id;
};

/**
 * Associates a reward function to each index.
 */
extern RewardFunction *all_reward_functions[];


/************************************
 *        REWARD FUNCTIONS          *
 ************************************/

/**
 * Returns 0.
 */
int get_no_reward(Game *game);

/**
 * Returns the number of lines removed in the last move.
 */
int get_reward_removed_lines(Game *game);

/**
 * Returns 1 if the game is not over.
 */
int get_reward_one(Game *game);

/**
 * Returns 1 if there was at least one removed line in the last move.
 */
int get_reward_at_least_one_line(Game *game);

/**
 * Returns a number which grows quickly with the number of removed lines. (useful for the RL competition)
 */
int get_reward_tetris_are_better(Game *game);

#endif
