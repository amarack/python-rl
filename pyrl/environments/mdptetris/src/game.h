/**
 * @defgroup game Game
 * @ingroup api
 * @brief A tetris game
 *
 * This module handles a game of Tetris. A game is composed by the board state,
 * the current piece and the score.
 *
 * @{
 */
#ifndef GAME_H
#define GAME_H

#include <stdio.h>
#include "types.h"
#include "board.h"
#include "piece.h"
#include "last_move_info.h"

/**
 * @brief Action decided by the player.
 *
 * An action is a decision made by the player in a given game state.
 * It is where the player drops the piece and in which orientation.
 */
struct Action {
  int orientation; /**< Rotation of the piece, from \c 0 to \c 3. */
  int column;      /**< Column where the piece is dropped, from \c 1 to \c board->width. */
};

/**
 * @brief Configuration of the game pieces.
 *
 * This structure describes the set of pieces to use in the game.
 * In standard Tetris, there are 7 pieces.
 */
typedef struct PieceConfiguration {
  int nb_pieces;            /**< Number of existing pieces (7 in standard Tetris). */
  Piece *pieces;            /**< Shapes of the existing pieces. */
  int *piece_sequence;      /**< The sequence of pieces falling
			     * (a NULL-terminated array of piece indexes),
			     * or NULL to choose the pieces randomly. */
  int nb_games;             /**< Number of games currently allocated that use this piece configuration. */
} PieceConfiguration;

/**
 * @brief A game.
 *
 * The game structure. The structure contains the game
 * configuration, the current state and some information
 * about the previous state of the game.
 */
struct Game {

  /**
   * @name Configuration of the pieces
   */
  PieceConfiguration *piece_configuration; /**< The pieces of Tetris. */

  int tetris_implementation;               /**< 0: Simplified Tetris, 1: RLC2008 Tetris, 2(to be done): Original Tetris */   
  /**
   * @name Game state
   */
  Board *board;                           /**< The wall state. */
  int game_over;                          /**< 1 if the game is over, 0 otherwise. */
  int score;                              /**< Number of lines removed since the beginning of the game. */
  Piece *current_piece;                   /**< The current piece falling. */
  int current_piece_index;                /**< Index of the current piece. */
  int current_piece_sequence_index;       /**< Current index in the sequence of pieces. */
  
  /**
   * @name Information about the previous state
   */
  int previous_piece_index;               /**< The last piece placed. */
  LastMoveInfo last_move_info;            /**< Information about the last move. */
};

/**
 * @name Game creation and destruction
 *
 * These functions allow to create or destroy a game.
 */
Game *new_game(int tetris_implementation, int width, int height, int allow_lines_after_overflow,
	       const char *pieces_file_name, int *piece_sequence);
Game *new_standard_game();
Game *new_game_from_parameters(CommonParameters *parameters);
Game *new_game_copy(const Game *other);
void free_game(Game *game);
/**
 * @}
 */

/**
 * @name Observation functions
 *
 * These functions provide some information about the current game state.
 * You can read directly the content of a game's structure.
 *
 * @{
 */
int game_get_nb_possible_orientations(Game *game);
int game_get_nb_possible_columns(Game *game, int orientation);
int game_get_current_piece(Game *game);
int game_get_nb_pieces(Game *game);
/**
 * @}
 */

/**
 * @name Modification functions
 *
 * These function change the current game state.
 * You should not change the content of a game's structure directly.
 *
 * @{
 */
int game_drop_piece(Game *game, const Action *action, int cancellable);
void game_cancel_last_move(Game *game);
void game_set_current_piece_index(Game *game, int piece_index);
void game_reset(Game *game);
void generate_next_piece(Game *game);
/**
 * @}
 */

/**
 * @name Displaying
 * @{
 */
void game_print(FILE *out, Game *game);
/**
 * @}
 */

#endif

/**
 * @}
 */
