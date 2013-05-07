/**
 * @defgroup game_statistics Game statistics
 * @ingroup api
 * @brief Statistics about some sequences (episodes) of games played
 *
 * This module handles the statistics about one or several sequences of games played.
 * A seqeuence of games is called an episode. Each episode is saved as a line
 * in the statistics file. The line contains the episode number, the best score,
 * the worst score, the mean score, the standard deviation and the scores of the
 * games played during the episode. Optionnaly, it can also contain the feature weights
 * if the games were played with a feature-based policy.
 *
 * As the first element of each line is the episode number and the second element is
 * the episode's mean score, the file can be plotted directly in Gnuplot.
 *
 * @{
 */
#ifndef GAMES_STATISTICS_H
#define GAMES_STATISTICS_H

#include <stdio.h>
#include "game.h"
#include "macros.h"

/**
 * @brief Statistics about the episodes.
 *
 * This structure contains all data about the current episode
 * and some global information about all episodes.
 */
struct GamesStatistics {
  /**
   * @name Statistics about an episode (reinitialized when games_statistics_reset() is called)
   */
  int *scores;               /**< Score of each game. */
  int nb_games_played;       /**< Number of games to play in the current episode. */
  int min_score;             /**< Worst score of a game in this episode. */
  int max_score;             /**< Best score of a game in this episode. */
  double mean;               /**< Mean score of the games in this episode. */
  double standard_deviation; /**< Standard deviation of the games in this episode. */

  /**
   * @name Information about all episodes played
   */
  int nb_episodes;           /**< Number of episodes done until now. */
  double best_mean;          /**< Best mean score of an episode. */
  FILE *stats_file;          /**< The file where the statistics are saved. */
};

/**
 * @name Statistics creation and destruction
 *
 * These functions allow to create or destroy a GameStatistics object.
 */
GamesStatistics *games_statistics_new(const char *stats_file_name, int nb_games, const char *comments);
void games_statistics_free(GamesStatistics *games_statistics);
/**
 * @}
 */

/**
 * @name Statistics update
 * 
 * These function update the statistics, taking new information into account.
 */
void games_statistics_add_game(GamesStatistics *stats, int score);
void games_statistics_end_episode(GamesStatistics *games_statistics, const FeaturePolicy *feature_policy);
/**
 * @}
 */

#endif

/**
 * @}
 */
