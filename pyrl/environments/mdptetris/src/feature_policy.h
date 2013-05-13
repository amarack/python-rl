/**
 * @defgroup feature_policy Feature policy
 * @ingroup api
 * @brief Feature handling
 *
 * This module provides general material to manipulate
 * a set of features. The feature functions themselves
 * are defined in the \ref feature_functions
 * "Feature functions" module.
 *
 * @see feature_functions
 * @{
 */
#ifndef FEATURE_POLICY_H
#define FEATURE_POLICY_H

#include "types.h"
#include "rewards.h"

#define NB_ORIGINAL_FEATURES 14

/**
 * @brief Constants to identify the features functions (or "basis functions").
 *
 * See the \ref feature_functions "Feature functions" module to have a precise
 * description of each feature.
 *
 * Note that our features are indexed with negative numbers.
 * Some of them are experimental and may perform not well...
 *
 * The features functions whose name starts with \c NEXT are designed to be called
 * several times during the same state evaluation, because they have several values to return
 * and a different weight is assigned to each one.
 * For example, feature #8 (\c NEXT_COLUMN_HEIGHT) should be
 * called 10 times (assuming the board width is 10), in order to get the height
 * of all columns. You can see that in the feature files (see for example
 * <code>features/bertsekas_initial.dat</code>: feature #8 appears 10 times).
 *
 * @see feature_functions
 */
typedef enum {

  /*
   * Our features.
   * When you add an original feature, update NB_ORIGINAL_FEATURES above
   */
  HOLE_DEPTHS = -1,                       /**< -1 (original feature): depth of each hole */
  ROWS_WITH_HOLES = -2,                   /**< -2 (original feature): number of rows with holes */
  NEXT_WALL_HEIGHT = -3,                  /**< -3 (original feature): there are \c h features where
					   * \c h is the board height: \c 1 for the top row of the
					   * wall, \c 0 elsewhere */
  SURROUNDED_HOLES = -4,                  /**< -4 (original feature): number of holes with full cells both
					   * below and above  */
  NEXT_LOCAL_VALUE_FUNCTION = -5,         /**< -5 (original feature): value function of the 5*5 board
					   * (96 features with the standard board size) */
  WELL_SUMS_FAST = -6,                    /**< -6 (original feature): our version of Dellacherie's feature #6 */
  DISTANCE_TO_TOP = -7,
  NEXT_WALL_DISTANCE_TO_TOP = -8,
  NEXT_COLUMN_HEIGHT_DIFFERENCE2 = -9,

  /* Square of features (for performance increase!) */
  DISTANCE_TO_TOP_SQUARE = -10,
  HOLE_DEPTHS_SQUARE = -11,
  HEIGHT_SQUARE = -12,
  NEXT_COLUMN_HEIGHT2 = -13,
  DIVERSITY = -14,

  /*
   * Special feature
   */
  CONSTANT = 0,                           /**< 0: constant term (useful for RL approaches) */

  /*
   * Features from Pierre Dellacherie (2003)
   */
  LANDING_HEIGHT,                         /**< 1 (Dellacherie): landing height */
  ERODED_PIECE_CELLS,                     /**< 2 (Dellacherie): eroded piece cells */
  ROW_TRANSITIONS,                        /**< 3 (Dellacherie): row transitions */
  COLUMN_TRANSITIONS,                     /**< 4 (Dellacherie): column transitions */
  HOLES,                                  /**< 5 (Dellacherie): holes */
  WELL_SUMS_DELLACHERIE,                  /**< 6 (Dellacherie): for each well, a value depending on the well depth */

  /*
   * Features from Bertsekas and Ioffe (1996)
   */
  WALL_HEIGHT,                            /**< 7 (Bertsekas): wall height */
  NEXT_COLUMN_HEIGHT,                     /**< 8 (Bertsekas): height of each column (\c w such features
					   * where \c w is the board width) */
  NEXT_COLUMN_HEIGHT_DIFFERENCE,          /**< 9 (Bertsekas): difference of height between adjacent
					   * columns (<code>w - 1</code> such features) */

  /*
   * Features from Colin Fahey (2006)
   */
  OCCUPIED_CELLS,                         /**< 10 (Fahey): number of occupied cells */
  WEIGHTED_CELLS,                         /**< 11 (Fahey): number of occupied cells, weighted by their height */
  WELLS,                                  /**< 12 (Fahey): depth of the wells */
  ROWS_ELIMINATED,                        /**< 13 (Fahey): number of lines removed in the last move */

  /*
   * Features from Bohm et al (2005) (not implemented yet)
   */
  ADJACENT_HOLES,                         /**< 14 (Bohm): adjacent holes */
  MAX_HEIGHT_DIFFERENCE,                  /**< 15 (Bohm): maximum height difference */
  MAX_WELL_DEPTH,                         /**< 16 (Bohm): maximum well depth */

  /*
   * Lagoudakis et al (2002) (not implemented yet)
   */
  HEIGHT_DIFFERENCES_SUM,                 /**< 17 (Lagoudakis): sum of the height differences */
  MEAN_HEIGHT,                            /**< 18 (Lagoudakis): mean column height */
  VARIATION_HEIGHT_DIFFERENCES_SUM,       /**< 19 (Lagoudakis): variation of \c HEIGHT_DIFFERENCES_SUM
					   * since the last move */
  VARIATION_MEAN_HEIGHT,                  /**< 20 (Lagoudakis): variation of \c MEAN_HEIGHT since the last move */
  VARIATION_WALL_HEIGHT,                  /**< 21 (Lagoudakis): variation of \c WALL_HEIGHT since the last move */
  VARIATION_HOLES                         /**< 22 (Lagoudakis): variation of \c HOLES since the last move */

} FeatureID;

/**
 * @brief A feature and its weight
 *
 * This structure defines a feature (i.e. a function to compute
 * a value depending on the game state) and a weight for this feature.
 */
struct Feature {
  FeatureFunction *get_feature_rating;    /**< The feature function. */
  double weight;                          /**< The weight associated to the feature. */
  FeatureID feature_id;                   /**< ID of the feature function. */
};

/**
 * @brief Policy defined from features.
 *
 * This structure describes a feature-based policy with
 * the features, their weights and the reward function.
 */
struct FeaturePolicy {

  /**
   * @name The features
   */
  Feature *features;                    /**< The features and their weights. */
  int nb_features;                      /**< Number of features used. */
  int update_column_heights_needed;     /**< 1 if the features need to compute the column heights */

  /**
   * @name Other policy settings
   */
  RewardDescription reward_description; /**< Immediate reward function used for the decision */
  int gameover_evaluation;              /**< 1 if the value of a gameover state is computed
					 * with the features, 0 if it is 0, -1 if it is -inf */
};

/**
 * @name Computation of features
 *
 * These functions use a feature policy to compute some information.
 *
 * @{
 */
double **get_policy_feature_ranges(Game *game, const FeaturePolicy *feature_policy);
double *get_feature_range(Game *game, FeatureID feature_id);

int contains_feature(const FeaturePolicy *feature_policy, FeatureID feature_id);
double evaluate_features(Game *game, const FeaturePolicy *feature_policy);
double *get_feature_values(Game *game, const FeaturePolicy *feature_policy);
void features_get_best_action(Game *game, const FeaturePolicy *feature_policy, Action *best_action);
/**
 * @}
 */

/**
 * @name Load or save a feature policy
 *
 * These functions load and save a feature policy in a file.
 *
 * @{
 */
void load_feature_policy(const char *feature_file_name, FeaturePolicy *feature_policy);
void save_feature_policy(const char *feature_file_name, const FeaturePolicy *feature_policy);
void create_feature_policy_original(int width, FeaturePolicy *feature_policy);
void create_feature_policy_dellacherie(int width, FeaturePolicy *feature_policy);

/**
 * @}
 */

/**
 * @name Playing games with a feature policy
 */
void feature_policy_play_game(const FeaturePolicy *feature_policy, Game *game);
double feature_policy_play_games(const FeaturePolicy *feature_policy, int nb_games, Game *game,
				 GamesStatistics *stats, int print_scores);
/**
 * @}
 */

/**
 * @name Displaying
 *
 * These functions display the features.
 *
 * @{
 */
void print_features(const FeaturePolicy *feature_policy);
void game_print_features(Game *game, const FeaturePolicy *feature_policy);
/**
 * @}
 */

#endif
/**
 * @}
 */
