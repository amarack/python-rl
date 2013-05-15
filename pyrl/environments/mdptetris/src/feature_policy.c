#include <stdlib.h>
#include <stdio.h>
#include "config.h"
#include "feature_policy.h"
#include "feature_functions.h"
#include "game.h"
#include "games_statistics.h"
#include "brick_masks.h"
#include "file_tools.h"
#include "macros.h"


double **get_policy_feature_ranges(Game *game, const FeaturePolicy *feature_policy) {
  double** theArray = NULL;
  double* range = NULL;
  int i;
    MALLOCN(theArray, double*, feature_policy->nb_features);

  for (i = 0; i < feature_policy->nb_features; i++) {
    MALLOCN(theArray[i], double, 2);
    range = get_feature_range(game, feature_policy->features[i].feature_id);
    theArray[i][0] = range[0];
    theArray[i][1] = range[1];
    FREE(range);
    }
  return theArray;

}

/* This function needs a lot of work, but these are good initial guesses */
double *get_feature_range(Game *game, FeatureID feature_id) {
  double *range = NULL;
  //range = (double *)malloc(2 * sizeof(double));
  //range[0] = 0.0;
  MALLOCN(range, double, 2);

  switch (feature_id) {
    case HOLE_DEPTHS:
      range[1] = (double)game->board->height;
      break;
    case ROWS_WITH_HOLES:
      range[1] = (double)game->board->height;
      break;
    case NEXT_WALL_HEIGHT:
      range[1] = (double)game->board->height;
      break;
    case SURROUNDED_HOLES:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case NEXT_LOCAL_VALUE_FUNCTION:
      range[1] = 1.0;
      break;
    case WELL_SUMS_FAST:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case DISTANCE_TO_TOP:
      range[1] = (double)game->board->height;
      break;
    case NEXT_WALL_DISTANCE_TO_TOP:
      range[1] = (double)game->board->height;
      break;
    case NEXT_COLUMN_HEIGHT_DIFFERENCE2:
      range[0] = -1 * (double)game->board->height;
      range[1] = (double)game->board->height;
      break;
    case DISTANCE_TO_TOP_SQUARE:
      range[1] = (double)(game->board->height*game->board->height);
      break;
    case HOLE_DEPTHS_SQUARE:
      range[1] = (double)(game->board->height*game->board->height);
      break;
    case HEIGHT_SQUARE:
      range[1] = (double)(game->board->height*game->board->height);
      break;
    case NEXT_COLUMN_HEIGHT2:
      range[1] = (double)(game->board->height*game->board->height);
      break;
    case DIVERSITY:
      range[1] = 1.0;
      break;
    case LANDING_HEIGHT:
      range[1] = (double)(game->board->height);
      break;
    case ERODED_PIECE_CELLS:
      range[1] = (double)(game->board->width * game->board->max_piece_height);
      break;
    case ROW_TRANSITIONS:
      range[1] = (double)(game->board->height * game->board->width);
      break;
    case COLUMN_TRANSITIONS:
      range[1] = (double)(game->board->height * game->board->width);
      break;
    case HOLES:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case WELL_SUMS_DELLACHERIE:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case WALL_HEIGHT:
      range[1] = (double)(game->board->height);
      break;
    case NEXT_COLUMN_HEIGHT:
      range[1] = (double)(game->board->height);
      break;
    case NEXT_COLUMN_HEIGHT_DIFFERENCE:
      range[0] = -1 * (double)game->board->height;
      range[1] = (double)game->board->height;
      break;
    case OCCUPIED_CELLS:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case WEIGHTED_CELLS:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case WELLS:
      range[1] = (double)(game->board->height * game->board->width)/2.0;
      break;
    case ROWS_ELIMINATED:
      range[1] = (double)(game->board->max_piece_height);
      break;
    case CONSTANT:
    default:
      range[0] = 1.0;
      range[1] = 1.0;
  }
  return range;
}

/**
 * @brief Returns whether or a feature policy contains a given feature.
 * @param feature_policy the features
 * @param feature_id id of the feature to search
 * @return 1 if this feature is present, 0 otherwise
 */
int contains_feature(const FeaturePolicy *feature_policy, FeatureID feature_id) {

  int i;
  int found = 0;
  for (i = 0; i < feature_policy->nb_features && !found; i++) {
    found = (feature_policy->features[i].feature_id == feature_id);
  }

  return found;
}

/**
 * @brief Evaluates a game state using a set of features.
 * @param game the current game state
 * @param feature_policy the feature-based policy
 * @return the evaluation of the state with the features
 */
double evaluate_features(Game *game, const FeaturePolicy *feature_policy) {
  double rating;
  int i, nb_features;
  Feature *feature;

  if (game->game_over && feature_policy->gameover_evaluation == 0) {
      rating = 0;
  }
  else if (game->game_over && feature_policy->gameover_evaluation == -1) {
      rating = -TETRIS_INFINITE;
  }
  else {               /* general case: evaluate with the features */
    if (feature_policy->update_column_heights_needed) {
      board_update_column_heights(game->board);
    }
    nb_features = feature_policy->nb_features;
    rating = 0;
    for (i = 0; i < nb_features; i++) {
      feature = &feature_policy->features[i];
      rating += feature->get_feature_rating(game) * feature->weight;
    }
  }
  return rating;
}

/**
 * @brief Returns the values of each feature in a state.
 *
 * The function returns an array with the values of each feature for the current state.
 * The game should not be over.
 *
 * @param game the current game state
 * @param feature_policy the feature-based policy
 * @return the value of each feature in this state
 */
double *get_feature_values(Game *game, const FeaturePolicy *feature_policy) {
  int i;
  double *feature_values;

  MALLOCN(feature_values, double, feature_policy->nb_features);

  if (feature_policy->update_column_heights_needed) {
    board_update_column_heights(game->board);
  }

/*   printf("\n---------\n"); */
/*   print_board(stdout, game->board); */

  for (i = 0; i < feature_policy->nb_features; i++) {
    feature_values[i] = feature_policy->features[i].get_feature_rating(game);
/*     printf("feature %d (id = %d): %f\n", i, feature_policy->features[i].feature_id, feature_values[i]); */
  }
/*   getchar(); */

  return feature_values;
}

/**
 * @brief Chooses the best action in a state.
 *
 * The action is chosen with respect to a feature set and a reward function.
 * Every action is tried, then each resulting state is evaluated.
 * This evaluation is added to the reward obtained. The action that
 * maximiz this value is selected.
 *
 * @param game the current game state
 * @param feature_policy the feature based policy
 * @param best_action pointer to store the best action found
 */
void features_get_best_action(Game *game, const FeaturePolicy *feature_policy, Action *best_action) {
  int nb_possible_orientations, nb_possible_columns, i, j;
  double evaluation, best_evaluation;
  Action action;

  best_evaluation = -TETRIS_INFINITE;
  best_action->orientation = 0;
  best_action->column = 1;

  /* try every possible action */
  nb_possible_orientations = game_get_nb_possible_orientations(game);
  for (i = 0; i < nb_possible_orientations; i++) {
    action.orientation = i;
    nb_possible_columns = game_get_nb_possible_columns(game, i);
    for (j = 1; j <= nb_possible_columns; j++) {
      action.column = j;

      /* make the action */
      game_drop_piece(game, &action, 1);

      /* compute immediate reward + evaluation of next state */

      /* debug
      if (feature_policy->reward_description.reward_function(game) != 0) {
	printf("oups: reward = %d\n", feature_policy->reward_description.reward_function(game));
      }
      */

      evaluation = feature_policy->reward_description.reward_function(game) + evaluate_features(game, feature_policy);

      /*board_print(stdout, game->board);*/
      /*game_print_features(game, feature_policy);*/
      /*printf("orientation %d, column %d: evaluation = %e\n", action.orientation, action.column, evaluation);*/



      if (DOUBLE_GREATER_THAN(evaluation,best_evaluation)) {
	best_evaluation = evaluation;
	best_action->orientation = i;
	best_action->column = j;
      }
      game_cancel_last_move(game);
    }
  }
}

/**
 * @brief Plays a game with a feature policy.
 * @param feature_policy the feature policy
 * @param game a game object (will be reseted at the beginning of the game)
 */
void feature_policy_play_game(const FeaturePolicy *feature_policy, Game *game) {

  Action action;
  game_reset(game);

  while (!game->game_over) {
    /* search the best move */
    features_get_best_action(game, feature_policy, &action); /* use the reward function of
								feature_policy to choose the move */

    /* play the best move found earlier */
    game_drop_piece(game, &action, 0);
  }
}

/**
 * @brief Play some games to evaluate a feature policy.
 * @param feature_policy the feature policy
 * @param nb_games number of games to play
 * @param game a game object (will be reseted at the beginning of each game)
 * @param stats a statistics object, to save the results (NULL to save nothing)
 * @param print_scores 1 to print the score of each game on the standard output, 0 otherwise
 * @return the mean score of the games
 */
double feature_policy_play_games(const FeaturePolicy *feature_policy, int nb_games, Game *game,
				 GamesStatistics *stats, int print_scores) {

  double mean_score;
  int i;

  mean_score = 0;
  for (i = 0; i < nb_games; i++) {
    feature_policy_play_game(feature_policy, game);
    if (print_scores) {
      printf("%d ", game->score);
      fflush(stdout);
    }

    if (stats != NULL) {
      games_statistics_add_game(stats, game->score);
    }
    else {
      mean_score += game->score;
    }
  }

  if (stats != NULL) {
    mean_score = stats->mean;
    games_statistics_end_episode(stats, feature_policy);
  }
  else {
    mean_score /= nb_games;
  }

  return mean_score;
}

void create_feature_policy_dellacherie(int width, FeaturePolicy *feature_policy) {
  int reward_function_id, gameover_evaluation;
  int i, nb_features, update_column_heights_needed;
  int feature_id;
  double weight;
  Feature *features;

  /* The rewards/gameover are not important for this, just using features */
  reward_function_id = 1;
  gameover_evaluation = 1;

  /* Number of features: 1 (constant) + 1 (wall height) + width (height of columns) + width (relative heights) */
  nb_features = 6;
  MALLOCN(features, Feature, nb_features);

  /* read each feature and its weight */
  update_column_heights_needed = 1;

  features[0].feature_id = LANDING_HEIGHT;
  features[1].feature_id = ERODED_PIECE_CELLS;
  features[2].feature_id = ROW_TRANSITIONS;
  features[3].feature_id = COLUMN_TRANSITIONS;
  features[4].feature_id = HOLES;
  features[5].feature_id = WELL_SUMS_DELLACHERIE;

  /* load the feature functions */
  for (i = 0; i < nb_features; i++) {
    features[i].weight = 0.0;
    feature_id = features[i].feature_id;
    features[i].get_feature_rating = feature_function(feature_id);
  }

  feature_policy->gameover_evaluation = gameover_evaluation;
  feature_policy->reward_description.reward_function_id = reward_function_id;
  feature_policy->reward_description.reward_function = all_reward_functions[reward_function_id];
  feature_policy->features = features;
  feature_policy->nb_features = nb_features;
  feature_policy->update_column_heights_needed = update_column_heights_needed;

  /* initialize the features system for these feature functions */
  features_initialize(feature_policy);
}




void create_feature_policy_original(int width, FeaturePolicy *feature_policy) {
  int reward_function_id, gameover_evaluation;
  int i, nb_features, update_column_heights_needed;
  int feature_id;
  double weight;
  Feature *features;

  /* The rewards/gameover are not important for this, just using features */
  reward_function_id = 1;
  gameover_evaluation = 1;

  /* Number of features: 1 (number of holes) + 1 (wall height) +
     width (height of columns) + width (relative heights) */
  nb_features = 2 + 2*width;

  MALLOCN(features, Feature, nb_features);
  // features = (Feature*) malloc(nb_features * sizeof(Feature));
  /* read each feature and its weight */
  update_column_heights_needed = 1;
  for (i = 0; i < width; i++) {
    features[i].feature_id = NEXT_COLUMN_HEIGHT_DIFFERENCE;
    features[i].weight = 0.0;
  }

  for (; i < 2*width; i++) {
    features[i].feature_id = NEXT_COLUMN_HEIGHT;
    features[i].weight = 0.0;
  }

  features[i].feature_id = HOLES;
  features[i].weight = 0.0;

  features[++i].feature_id = WALL_HEIGHT;
  features[i].weight = 0.0;

  /* load the feature functions */
  for (i = 0; i < nb_features; i++) {
    feature_id = features[i].feature_id;
    features[i].get_feature_rating = feature_function(feature_id);
  }

  feature_policy->gameover_evaluation = gameover_evaluation;
  feature_policy->reward_description.reward_function_id = reward_function_id;
  feature_policy->reward_description.reward_function = all_reward_functions[reward_function_id];
  feature_policy->features = features;
  feature_policy->nb_features = nb_features;
  feature_policy->update_column_heights_needed = update_column_heights_needed;

  /* initialize the features system for these feature functions */
  features_initialize(feature_policy);
}



/**
 * @brief Loads the feature based policy and the initial weights from a file.
 * @param feature_file_name file to read (this file will be searched in the
 * current directory and then in the data directory)
 * @param feature_policy object to store the information read
 * @see save_feature_policy()
 */
void load_feature_policy(const char *feature_file_name, FeaturePolicy *feature_policy) {
  FILE *feature_file;
  int reward_function_id, gameover_evaluation;
  int i, nb_features, update_column_heights_needed;
  int feature_id;
  double weight;
  Feature *features;

  feature_file = open_data_file(feature_file_name, "r");

  if (feature_file == NULL) {
    DIE1("Cannot read the feature file '%s'", feature_file_name);
  }

  /* read the reward function id */
  FSCANF(feature_file, "%d", &reward_function_id);

  /* read the gameover evaluation value */
  FSCANF(feature_file, "%d", &gameover_evaluation);


  /* read the number of features */
  FSCANF(feature_file, "%d", &nb_features);
  MALLOCN(features, Feature, nb_features);

  /* read each feature and its weight */
  update_column_heights_needed = 0;
  for (i = 0; i < nb_features; i++) {
    FSCANF2(feature_file, "%d %lf", &feature_id, &weight);
    features[i].feature_id = feature_id;
    features[i].weight = weight;

    switch (feature_id) {
    case NEXT_COLUMN_HEIGHT:
    case NEXT_COLUMN_HEIGHT_DIFFERENCE:
    case MAX_HEIGHT_DIFFERENCE:
    case HEIGHT_DIFFERENCES_SUM:
    case MEAN_HEIGHT:
    case VARIATION_HEIGHT_DIFFERENCES_SUM:
    case VARIATION_MEAN_HEIGHT:
    case VARIATION_WALL_HEIGHT:
    case NEXT_WALL_DISTANCE_TO_TOP:
    case NEXT_COLUMN_HEIGHT_DIFFERENCE2:
    case NEXT_COLUMN_HEIGHT2:
    case DIVERSITY:
      update_column_heights_needed = 1;
    }
  }

  fclose(feature_file);

  /* load the feature functions */
  for (i = 0; i < nb_features; i++) {
    feature_id = features[i].feature_id;
    features[i].get_feature_rating = feature_function(feature_id);
  }

  feature_policy->gameover_evaluation = gameover_evaluation;
  feature_policy->reward_description.reward_function_id = reward_function_id;
  feature_policy->reward_description.reward_function = all_reward_functions[reward_function_id];
  feature_policy->features = features;
  feature_policy->nb_features = nb_features;
  feature_policy->update_column_heights_needed = update_column_heights_needed;

  /* initialize the features system for these feature functions */
  features_initialize(feature_policy);
}

/**
 * @brief Saves the feature based policy and their weights into a file.
 * @param feature_file_name the file to write (the file name is relative to current directory)
 * @param feature_policy the feature_policy to save into the file
 * @see load_feature_policy()
 */
void save_feature_policy(const char *feature_file_name, const FeaturePolicy *feature_policy) {
  FILE *feature_file;
  int i;

  feature_file = fopen(feature_file_name, "w");

  if (feature_file == NULL) {
    DIE1("Unable to write the feature file '%s'", feature_file_name);
  }

  /* write the reward function id */
  fprintf(feature_file, "%d\n", feature_policy->reward_description.reward_function_id);

  /* write the gameover evaluation value */
  fprintf(feature_file, "%d\n", feature_policy->gameover_evaluation);

  /* write the number of features */
  fprintf(feature_file, "%d\n", feature_policy->nb_features);

  /* write each feature and its weight */
  for (i = 0; i < feature_policy->nb_features; i++) {
    fprintf(feature_file, "%d %e\n", feature_policy->features[i].feature_id, feature_policy->features[i].weight);
  }

  fclose(feature_file);
}

/**
 * @brief Displays the weights of the features.
 *
 * A feature is represented on a line with two values: the feature id and the weight.
 *
 * @param feature_policy the feature policy
 */
void print_features(const FeaturePolicy *feature_policy) {
  int i;

  for (i = 0; i < feature_policy->nb_features; i++) {
    printf("%d->%f ", feature_policy->features[i].feature_id, feature_policy->features[i].weight);
  }
  printf("\n");
}

/**
 * @brief Displays the values of the features in a given state.
 *
 * This function is useful for debugging/watching.
 *
 * @param game the current game state
 * @param feature_policy the feature policy to use
 */
void game_print_features(Game *game, const FeaturePolicy *feature_policy) {
  int i;
  double *feature_values;

  printf("Features: ");

  feature_values = get_feature_values(game, feature_policy);

  for (i = 0; i < feature_policy->nb_features; i++) {
    printf("%i->%f ",feature_policy->features[i].feature_id , feature_values[i]);
  }
  printf("=> Value: %f\n", evaluate_features(game, feature_policy));

  FREE(feature_values);
}
