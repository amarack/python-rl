#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include "config.h"
#include "games_statistics.h"
#include "feature_policy.h"
#include "macros.h"

/*
 * Private functions.
 */
static void parse_previous_statistics(GamesStatistics *games_statistics);
static void games_statistics_reset(GamesStatistics *games_statistics);

/**
 * @brief Creates a GamesStatistics object and initializes it.
 *
 * @param stats_file_name name of a file where you want the statistics to be saved
 * (can be a NULL to save nothing)
 * @param nb_games maximum number of games that will be played in each episode
 * @param comments a comment to add at the beginning of the file (can be NULL)
 * @see games_statistics_free()
 */
GamesStatistics *games_statistics_new(const char *stats_file_name, int nb_games, const char *comments) {

  GamesStatistics *statistics;

  MALLOC(statistics, GamesStatistics);
  CALLOC(statistics->scores, int, nb_games);

  statistics->nb_episodes = 0;
  statistics->best_mean = 0;
  statistics->stats_file = NULL;

  if (stats_file_name != NULL) {
    /* a file was specified, so the statistics will be saved there */

    statistics->stats_file = fopen(stats_file_name, "r");

    if (statistics->stats_file == NULL) {
      /* this is a new file: open in writing */

      statistics->stats_file = fopen(stats_file_name, "w");

      if (statistics->stats_file == NULL) {
	DIE1("Cannot create the statistics file '%s'", stats_file_name);
      }

      if (comments != NULL) {
	/* add some comments on the first line */
	fprintf(statistics->stats_file, "# %s\n", comments);
      }

      fprintf(statistics->stats_file, "# Episode\tMean score\tMin\tMax\tStandard deviation\tScores\t# Feature weights\t\n");
      fflush(statistics->stats_file);
    }
    else {
      /* this is an existing file: first we parse it, then and we will continue its statistics */

      parse_previous_statistics(statistics);
      fclose(statistics->stats_file);
      statistics->stats_file = fopen(stats_file_name, "a");
    }
  }

  /* initialize the first episode */
  games_statistics_reset(statistics);

  return statistics;
}

/**
 * @brief Destroys a GamesStatistics object.
 * @param games_statistics the object to destroy
 * @see games_statistics_new()
 */
void games_statistics_free(GamesStatistics *games_statistics) {

  /* close the file if any */
  if (games_statistics->stats_file != NULL) {
    fclose(games_statistics->stats_file);
  }

  FREE(games_statistics->scores);
  FREE(games_statistics);
}

/**
 * @brief Initializes the statistics with some previous statistics saved.
 *
 * When the statistics file already exists, this function is called and
 * the statistics of this file are continued.
 *
 * @param games_statistics the statistics
 */
static void parse_previous_statistics(GamesStatistics *games_statistics) {

  FILE *stats_file = games_statistics->stats_file;
  char *line = NULL;
  char *field_episode = NULL;
  char *field_mean;
  double mean;
  size_t n, c;

  while (!feof(stats_file)) {

    c = getline(&line, &n, stats_file);

    if (n > 1 && line[0] != '#') {

      field_episode = strtok(line, " \t");
      field_mean = strtok(NULL, " \t");
      if (field_mean != NULL) {
	
	mean = atof(field_mean);
	if (mean > games_statistics->best_mean) {
	  games_statistics->best_mean = mean;
	}
	games_statistics->nb_episodes++;
      }
    }
  }

  free(line);
}

/**
 * @brief Resets the statistics of the current episode.
 *
 * This function is called when an episode begins.
 *
 * @param games_statistics the statistics to reset
 */
static void games_statistics_reset(GamesStatistics *games_statistics) {
  games_statistics->nb_games_played = 0;
  games_statistics->min_score = -1;
  games_statistics->max_score = 0;
  games_statistics->mean = 0.0;
  games_statistics->standard_deviation = 0.0;
}

/**
 * @brief Updates the statistics of the current episode, adding a new game.
 *
 * @param stats the statistics to update
 * @param score score of the game to add
 */
void games_statistics_add_game(GamesStatistics *stats, int score) {

  /* update the score list */
  stats->scores[stats->nb_games_played] = score;
  
  /* update the maximum score */
  if (score > stats->max_score) {
    stats->max_score = score;
  }
  
  /* update the minimum score */
  if (score < stats->min_score || stats->min_score == -1) {
    stats->min_score = score;
  }
  
  /* update the mean score */
  stats->mean = (stats->mean * stats->nb_games_played + score) / (stats->nb_games_played + 1);
  
  /* update the number of games played */
  stats->nb_games_played++;
  
  /* update the standard deviation */
  stats->standard_deviation = 0.0;/*gsl_stats_int_sd_m(stats->scores, 1, stats->nb_games_played, stats->mean);*/

}

/**
 * @brief Ends the current episode.
 *
 * This function has to be called when all games of the episode are played.
 * The statistics of the episode are printed into a line of the file if it was not NULL.
 * The statistics are then reseted for the next episode, and nb_episodes is incremented.
 *
 * @param games_statistics the statistics
 * @param feature_policy the features used to play the games in this episode
 * (can be NULL if your policy wasn't based on features)
 */
void games_statistics_end_episode(GamesStatistics *games_statistics, const FeaturePolicy *feature_policy) {
  int i;

  if (games_statistics->stats_file != NULL) {

    /* print the statistics */
    fprintf(games_statistics->stats_file, "%d\t\t%f\t%d\t%d\t%f\t\t",
	    games_statistics->nb_episodes,
	    games_statistics->mean,
	    games_statistics->min_score,
	    games_statistics->max_score,
	    games_statistics->standard_deviation);

    /* print the game scores */
    for (i = 0; i < games_statistics->nb_games_played; i++) {
      fprintf(games_statistics->stats_file, "%d ", games_statistics->scores[i]);
    }

    /* print the feature weights if any */
    if (feature_policy != NULL) {
      fprintf(games_statistics->stats_file, "\t# ");
      for (i = 0; i < feature_policy->nb_features; i++) {
	fprintf(games_statistics->stats_file, "%f ", feature_policy->features[i].weight);
      }
    }

    fprintf(games_statistics->stats_file, "\n");

    fflush(games_statistics->stats_file);
  }

  /* update nb_episodes and best_mean */
  games_statistics->nb_episodes++;
  if (games_statistics->mean > games_statistics->best_mean) {
    games_statistics->best_mean = games_statistics->mean;
  }

  /* initialize the next episode */
  games_statistics_reset(games_statistics);
}
