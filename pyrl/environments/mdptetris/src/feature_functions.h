/**
 * @defgroup feature_functions Feature functions
 * @ingroup api
 * @brief Definition of the features functions.
 *
 * This module contains the definition of all feature functions.
 *
 * @see feature_policy
 * @{
 */
#ifndef FEATURE_FUNCTIONS_H
#define FEATURE_FUNCTIONS_H

#include "types.h"
#include "feature_policy.h"

/**
 * @name General feature function handling
 * @{
 */
FeatureFunction *feature_function(FeatureID feature_id);
void features_initialize(const FeaturePolicy *feature_policy);
void features_exit(void);
/**
 * @}
 */

/**
 * @name Special feature functions
 * @{
 */
double get_constant(Game *game);
/**
 * @}
 */

/**
 * @name Original feature functions
 */
double get_hole_depths(Game *game);
double get_rows_with_holes(Game *game);
double get_next_wall_height(Game *game);
double get_surrounded_holes(Game *game);
double get_next_local_value_function(Game *game);
double get_well_sums_fast(Game *game);

double get_wall_distance_to_top(Game *game);
double get_next_column_distance_to_top(Game *game);
double get_next_column_height_difference2(Game *game);

double get_wall_distance_to_top_square(Game *game);
double get_hole_depths_square(Game *game);
double get_height_square(Game *game);
double get_next_column_height2(Game *game);

double get_diversity(Game *game);

/**
 * @}
 */

/**
 * @name Feature functions from Dellacherie (2003)
 * @{
 */
double get_landing_height(Game *game);
double get_eroded_piece_cells(Game *game);
double get_row_transitions(Game *game);
double get_column_transitions(Game *game);
double get_holes(Game *game);
double get_well_sums_dellacherie(Game *game);
/**
 * @}
 */

/**
 * @name Feature functions from Bertsekas and Ioffe (1996)
 */
double get_wall_height(Game *game);
double get_next_column_height(Game *game);
double get_next_column_height_difference(Game *game);
/**
 * @}
 */

/**
 * @name Feature functions from Fahey (2006)
 */
double get_occupied_cells(Game *game);
double get_weighted_cells(Game *game);
double get_wells(Game *game);
double get_rows_eliminated(Game *game);
/**
 * @}
 */

/**
 * @name Feature functions from Bohm et al. (2005)
 */
double get_max_height_difference(Game *game);
/**
 * @}
 */

#endif

/**
 * @}
 */
