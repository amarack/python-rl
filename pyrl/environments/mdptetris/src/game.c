#include <stdlib.h>
#include <stdio.h>
#include "config.h"
#include "game.h"
#include "macros.h"
#include "board.h"
#include "piece.h"
#include "common_parameters.h"
#include "random.h"

/*
 * Private functions.
 */
static void restore_previous_piece(Game *game);

/**
 * @brief Creates a new tetris game.
 *
 * @param width board width (10 in standard Tetris; must be lower than or equal to 14)
 * @param height board height (20 in standard Tetris)
 * @param allow_lines_after_overflow 1 to enable the lines completion when the piece overflows
 * @param pieces_file_name a file describing the pieces (pieces4.dat for standard Tetris)
 * @param piece_sequence a predeterminated sequence of pieces, or NULL to generate the pieces randomly;
 * the sequence of pieces must be a NULL-terminated array of pieces indexes
 * @return the game created
 * @see new_standard_game(), new_game_from_parameters(), new_game_copy(), free_game()
 */
Game *new_game(int tetris_implementation, int width, int height, int allow_lines_after_overflow, const char *pieces_file_name, int *piece_sequence) {
  Game *game;

  MALLOC(game, Game);
  MALLOC(game->piece_configuration, PieceConfiguration);
  load_pieces(pieces_file_name, &game->piece_configuration->nb_pieces, &game->piece_configuration->pieces);
  game->tetris_implementation = tetris_implementation;
  game->board = new_board(width, height, allow_lines_after_overflow,
			  game->piece_configuration->nb_pieces, game->piece_configuration->pieces);
  game->piece_configuration->piece_sequence = piece_sequence;
  game->piece_configuration->nb_games = 1;
  game_reset(game);

  return game;
}

/**
 * @brief Creates a new standard Tetris game.
 * 
 * The standard configuration is: 20 rows, 10 columns and the 7 standard pieces.
 * The pieces are loaded from the file \c pieces4.dat,
 * so this file is supposed to be in the current directory.
 *
 * @return the game created
 * @see new_game(), new_game_from_parameters(), new_game_copy(), free_game()
 */
Game *new_standard_game() {

  return new_game(0, 10, 20, 0, "pieces4.dat", NULL);

}


int game_get_nb_pieces(Game *game) {

  return game->piece_configuration->nb_pieces;
}
/**
 * @brief Creates a game with somes parameters given in a structure CommonParameters.
 *
 * The structure CommonParameters specifies the board and the pieces.
 *
 * @param parameters the parameters
 * @return the game created
 * @see new_game(), new_standard_game(), new_game_copy(), free_game()
 */
Game *new_game_from_parameters(CommonParameters *parameters) {
  return new_game(parameters->tetris_implementation, 
		  parameters->board_width,
		  parameters->board_height,
		  parameters->allow_lines_after_overflow,
		  parameters->piece_file_name,
		  NULL);
}

/**
 * @brief Creates a copy of a game.
 * @return the game created
 * @see new_game(), new_standard_game(), new_game_from_parameters(), free_game()
 */
Game *new_game_copy(const Game *other) {
  
  Game *game;
  MALLOC(game, Game);
  *game = *other;

  /* deep copy of the board */
  game->tetris_implementation=other->tetris_implementation;
  game->board = new_board_copy(other->board);
  game->piece_configuration->nb_games++;

  return game;
}

/**
 * @brief Destroys a game.
 * @param game the game to destroy
 * @see new_game(), new_standard_game(), new_game_from_parameters(), new_game_copy()
 */
void free_game(Game *game) {
  int i;

  game->piece_configuration->nb_games--;
  if (game->piece_configuration->nb_games == 0) {
    /* free the piece configuration if it is not used anymore */
    for (i = 0; i < game->piece_configuration->nb_pieces; i++) {
      free_piece(&game->piece_configuration->pieces[i]);
    }
    FREE(game->piece_configuration->pieces);
    FREE(game->piece_configuration);
  }

  free_board(game->board);
  FREE(game);
}

/**
 * @brief Returns the number of possible orientations for the current piece.
 *
 * @param game the game
 * @return the number of possible orientations for the current piece
 * @see game_get_nb_possible_columns()
 */
int game_get_nb_possible_orientations(Game *game) {
  return game->current_piece->nb_orientations;
}


int game_get_current_piece(Game *game) {
  return game->current_piece_index;
}

/**
 * @brief Returns the number of columns where you can put the current piece.
 *
 * This number of columns depends on the orientation of the piece.
 *
 * @param game the game
 * @param orientation orientation of the current piece: <code>0 <= orientation <
 * game_get_nb_possible_orientations(game)</code>
 * @return the number of authorized columns for the current piece
 * @see game_get_nb_possible_orientations()
 */
int game_get_nb_possible_columns(Game *game, int orientation) {

  /* debug (disabled for performance reasons)
  if (orientation < 0 || orientation >= game_get_nb_possible_orientations(game)) {
    game_print(stdout, game);
    DIE1("Illegal orientation: %d\n", orientation);
  }
  */

  return game->board->width - game->current_piece->orientations[orientation].width + 1;
}

/**
 * @brief Changes the current piece of the game.
 *
 * @param game a game
 * @param piece_index index of the new current piece: <code>0 <= piece_index <
 * game->piece_configuration.nb_pieces</code>
 */
void game_set_current_piece_index(Game *game, int piece_index) {
  game->current_piece_index = piece_index;
  game->current_piece = &game->piece_configuration->pieces[piece_index];
}

/**
 * @brief Generates the next piece.
 *
 * This function is called when the current piece is placed.
 * If there is a sequence of pieces (i.e. <code>game->piece_sequence != NULL</code>),
 * the next piece is picked from this sequence, otherwise it is chosen randomly.
 *
 * @param game the game
 * @see restore_previous_piece()
 */
void generate_next_piece(Game *game) {
  int piece_index;
  PieceConfiguration *piece_configuration;

  piece_configuration = game->piece_configuration;

  if (piece_configuration->piece_sequence == NULL) {
    /* the pieces are generated randomly */
    piece_index = random_uniform(0, piece_configuration->nb_pieces);
  }
  else {
    /* the pieces are generated from a sequence */

/*     printf("generating a new piece\n"); */
/*     printf("current piece: %d\n", game->current_piece_index); */
/*     if (game->current_piece == NULL) { */
/*       printf("current_piece is NULL\n"); */
/*     } */
/*     else { */
/*       print_piece_orientation(stdout, &game->current_piece->orientations[0]); */
/*     } */

    game->current_piece_sequence_index++;
    piece_index = piece_configuration->piece_sequence[game->current_piece_sequence_index];

    if (piece_index == -1) {
      /* the sequence is finished, let's restart it */
      game->current_piece_sequence_index = 0;
      piece_index = piece_configuration->piece_sequence[0];
    }

/*     printf("new piece: %d\n", piece_index); */
/*     print_piece_orientation(stdout, &game->configuration.pieces[piece_index].orientations[0]); */
  }
  
  game_set_current_piece_index(game, piece_index);

/*   printf("Done. current piece is now: %d\n", game->current_piece_index); */
/*   print_piece_orientation(stdout, &game->current_piece->orientations[0]); */
}

/**
 * @brief Restores the previous piece.
 *
 * This function is the opposite of generate_next_piece().
 *
 * @param game the game
 * @see generate_next_piece()
 */
static void restore_previous_piece(Game *game) {
  int i;
  PieceConfiguration *piece_configuration;

  piece_configuration = game->piece_configuration;

  game->current_piece_index = game->previous_piece_index;
  game->current_piece = &piece_configuration->pieces[game->current_piece_index];

  if (piece_configuration->piece_sequence != NULL) {
    game->current_piece_sequence_index--;
    if (game->current_piece_sequence_index == -1) {
      i = 1;
      while (piece_configuration->piece_sequence[i] != -1) {
	i++;
      }
      game->current_piece_sequence_index = i - 1;
    }
  } 
}

/**
 * @brief Makes a move.
 *
 * Drops the current piece in a column and scores the points for the removed lines.
 * The move is supposed to be authorized, i.e. <code>0 <= action->orientation <
 * game_get_nb_possible_orientations(game)</code> and <code>0 < action->column <=
 * game_get_nb_possible_columns(game, action->orientation)</code>.
 * Otherwise the result of this function is undetermined.
 * The game should not be over before this function is called, but it might be over after this move.
 *
 * @param game the game
 * @param action orientation and column of the piece (for the column, the left column of the piece is considered)
 * @param cancellable true to allow the move to be cancelled later
 * @return the number of lines just removed
 * @see game_cancel_last_move()
 */
int game_drop_piece(Game *game, const Action *action, int cancellable) {

/*   int piecenb,index=0; */
/*   PieceConfiguration *piece_configuration; */
/*   int orientation;   */
/*   uint16_t *board_rows; */
/*   uint16_t *piece_bricks; */
/*   int piece_height, piece_width; */
/*   int collision; */
/*   int column; */
/*   int y,i; */
/*   int current_row; */
/*   Piece *piece; */
/*   PieceOrientation *oriented_piece; */
  int removed_lines;

  /* ensure the game is not over */
  if (game->game_over) {
    DIE("Trying to make a move but the game is over");
  }

  /* update the board (choose different functions depending on the type of implementation) */
  switch(game->tetris_implementation) {   

  case 0: /* SIMPLIFIED */
    removed_lines = board_drop_piece(game->board, &game->current_piece->orientations[action->orientation],
				     action->orientation, action->column,
				     &game->last_move_info, cancellable);
    break;
  case 1: /* RLC */
    removed_lines = board_drop_piece_rlc(game->board, game->piece_configuration->pieces, game->current_piece_index,
						 action->orientation, action->column,
						 &game->last_move_info, cancellable);
    break;
  default:
    DIE("This should not happen (game_drop_piece)");
    break;
  }

  game->previous_piece_index = game->current_piece_index;

  /* update the score*/
  game->score += removed_lines;

  switch(game->tetris_implementation) {
  
  case 0: /* SIMPLIFIED */
    /* check game over */
    if (game->board->wall_height > game->board->height) {
      game->game_over = 1;
    }
    else {
      /* generate the next piece */
      generate_next_piece(game);
    }
    break;
  
  case 1: /* RLC */
    if (game->board->wall_height > game->board->height-2) {
      game->game_over = 1;
      game->last_move_info.nb_steps+=2; 
    } else {
      /* generate the next piece */
      generate_next_piece(game);
    }
    break;
  

    /* ??? old complicated RLC implementation */
/*     piece_configuration = game->piece_configuration; */

     /* store the current piece */ 
/*     piecenb = game->current_piece_index; */
/*     if (piece_configuration->piece_sequence != NULL) { */
/*       index=game->current_piece_sequence_index; */
/*     } */
     /* we need to generate a new piece */ 
/*     generate_next_piece(game); */
        
     /* Test whether the new piece collides or lays on something */         
/*     piece=game->current_piece; */
/*     board_rows = game->board->rows;     */
/*     orientation = piece->nb_orientations/2; */
/*     oriented_piece = &piece->orientations[orientation]; */
/*     piece_bricks = oriented_piece->bricks; */
/*     piece_height = oriented_piece->height; */
/*     piece_width = oriented_piece->width; */
    /*     column = (game->board->width)/2; */ /* at the beginning: put the piece at the center */ 
    /*     if (game->current_piece_index==0) {*/ /* special treatment for piece I */ 
/*       column--;  */
/*     } */
      /*     y = game->board->height-piece_height; */ /* height coordinate of the piece at the beginning of the fall (the top of the piece touches the top of the board */       
/*     collision = 0;     */
      /*     current_row=y; */ /* can it be inserted ? */ 
/*     for (i = 0; i < piece_height && !collision; i++, current_row++) { */
/*       collision = board_rows[current_row] & (piece_bricks[i] >> column);       */
/*     } */
      /*     current_row=y-1; */ /* does it lay on something ? */ 
/*     for (i = 0; i < piece_height && !collision; i++, current_row++) { */
/*       collision = board_rows[current_row] & (piece_bricks[i] >> column);       */
/*     } */

     /* It if does: it is game over and we restore the previous piece!  */
/*     if (collision) { */

/*       game->game_over = 1;        */

       /* restore the previous piece when game over is observed */         
/*       game->current_piece_index = piecenb; */
/*       game->current_piece = &piece_configuration->pieces[piecenb]; */
      /*       if (piece_configuration->piece_sequence != NULL) {*/  /* need to rewind */ 
/* 	game->current_piece_sequence_index = index; */
/*       }      */
/*     } */
/*    break;*/
  }
    

  return removed_lines;
}




/**
 * @brief Cancels the last move.
 *
 * The last dropped piece is removed and the previous game state is restored.
 * This is possible only if \c cancellable was set to \c 1 when you called
 * game_drop_piece().
 *
 * @param game the game
 * @see game_drop_piece()
 */
void game_cancel_last_move(Game *game) {
  game->score -= game->last_move_info.removed_lines;
  if (game->game_over) {
    game->game_over = 0;
  }
  else {
    restore_previous_piece(game);
  }
  board_cancel_last_move(game->board);
}

/**
 * @brief Resets the game and starts a new one.
 *
 * If you have many games to play, you should call this function
 * once a game is over, instead of allocating a new one.
 *
 * @param game the game to reset
 */
void game_reset(Game *game) {
  board_reset(game->board);
  game->game_over = 0;
  game->score = 0;
  game->current_piece_sequence_index = -1;
  game->last_move_info.removed_lines = 0;
  game->last_move_info.landing_height_bottom = 0;
  game->last_move_info.eliminated_bricks_in_last_piece = 0;
  game->last_move_info.oriented_piece = NULL;
  generate_next_piece(game);
}

/**
 * @brief Prints a human-readable view of the current state in a file.
 *
 * The current piece, a view of the board and the game score are printed.
 *
 * @param out file to write, e.g. stdout
 * @param game the game
 */
void game_print(FILE *out, Game *game) {
  /* print the game pieces 
     int i;
     fprintf(out, "Game pieces:\n");
     for (i = 0; i < game->nb_pieces; i++) {
     fprintf(out, "  Piece %d\n", i);
     print_piece_orientations(out, &game->pieces[i]);
     fprintf(out, "\n");
     }
  */

  fprintf(out, "Current piece:\n");
  piece_print(out, game->current_piece); 
  board_print(out, game->board);
   
/*   print_last_move_info(out, &game->last_move_info); */
  fprintf(out, "Game score: %d\n", game->score);
}
