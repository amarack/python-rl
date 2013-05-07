#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "config.h"
#include "macros.h"
#include "board.h"
#include "piece.h"
#include "brick_masks.h"
#include <math.h>

/**
 * @brief Creates a new empty board.
 *
 * @param width board width (10 in standard Tetris; must be lower than or equal to 14)
 * @param height board height (20 in standard Tetris)
 * @param allow_lines_after_overflow 1 to enable the lines completion when the piece overflows
 * @param nb_pieces number of existing pieces (7 for standard Tetris)
 * @param pieces array of existing pieces (\c nb_pieces elements)
 * @return the board created
 * @see free_board()
 */
Board *new_board(int width, int height, int allow_lines_after_overflow, int nb_pieces, Piece *pieces) {
  Board *board;
  int i, j;

  MALLOC(board, Board);
  board->width = width;
  board->height = height;
  board->allow_lines_after_overflow = allow_lines_after_overflow;
  CALLOC(board->column_heights, int, width + 1);

  /* Compute an empty row, for example 1000000000011111 for standard Tetris.
   * Note that an empty row is not really empty because of the side borders.
   */
  board->empty_row = brick_masks[0] | brick_masks[width+1];
  for (i = width + 2; i < 16; i++) {
    board->empty_row |= brick_masks[i];
  }
  board->full_row = 0xFFFF;
  
  /* compute max_piece_height (maximum possible height of a piece) */
  board->max_piece_height = 0;
  for (i = 0; i < nb_pieces; i++) {
    for (j = 0; j < pieces[i].nb_orientations; j++) {
      board->max_piece_height = MAX(board->max_piece_height, pieces[i].orientations[j].height);
    }
  }

  board->extended_height = height + board->max_piece_height;

  CALLOC(board->rows, uint16_t, board->extended_height);
  CALLOC(board->previous_rows, uint16_t, board->extended_height);

  /* make the rows empty */
  board_reset(board);

  return board;
}

/**
 * @brief Creates a copy of a board.
 */
Board *new_board_copy(const Board *other) {
  
  Board *board;
  MALLOC(board, Board);
  *board = *other;

  CALLOC(board->rows, uint16_t, board->extended_height);
  MEMCPY(board->rows, other->rows, uint16_t, board->extended_height);

  CALLOC(board->previous_rows, uint16_t, board->extended_height);
  MEMCPY(board->previous_rows, other->previous_rows, uint16_t, board->extended_height);

  CALLOC(board->column_heights, int, board->width + 1);
  MEMCPY(board->column_heights, other->column_heights, int, board->width + 1);

  return board;
}

/**
 * @brief Destroys a board.
 *
 * @param board the board to destroy
 * @see new_board()
 */
void free_board(Board *board) {
  FREE(board->rows);
  FREE(board->previous_rows);
  FREE(board->column_heights);
  FREE(board);
}

/**
 * @brief Drops a new piece onto the wall.
 * 
 * This function is called by game_drop_piece().
 * This is where the simulations spend most of the time, so the code is optimized
 * as much as we could. Each row is treated globally (there is no loop on the columns).
 * The move is supposed to be authorized, otherwise the result of this function is undetermined.
 *
 * @param board the board
 * @param oriented_piece the oriented piece to add
 * @param orientation the piece orientation (used to update \c last_move_info)
 * @param column where to drop the piece (the left column of the piece is considered),
 * the first column is \c 1
 * @param last_move_info if not \c NULL, some information about this move will be stored here
 * (useful for some of the features)
 * @param cancellable true to allow the move to be cancelled later
 * @return the number of removed lines
 * @see board_cancel_last_move()
 */
int board_drop_piece(Board *board, PieceOrientation *oriented_piece, int orientation, int column,
		     LastMoveInfo *last_move_info, int cancellable) {
  int i, i_stop, j;
  int destination;              /* index of the row where the bottom part of the piece is put */
  int destination_top;          /* 1 + (index of the highest row occupied by the piece once it is put) */           
  uint16_t *board_rows;
  uint16_t *piece_bricks;
  uint16_t empty_row, full_row;
  int collision;
  int piece_height, piece_width;
  int current_row;
  int removed_lines;
  int wall_height;

  board_rows = board->rows;
  empty_row = board->empty_row;
  full_row = board->full_row;
  piece_bricks = oriented_piece->bricks;
  wall_height = board->wall_height;
  removed_lines = 0;

  /* backup the board if necessary */
  if (cancellable) {
    MEMCPY(board->previous_rows, board_rows, uint16_t, board->extended_height);
    board->previous_wall_height = wall_height;
  }

  /* initialize last_move_info */
  if (last_move_info != NULL) {
    last_move_info->eliminated_bricks_in_last_piece = 0;
  }
  
  /* we will search the lowest available row for the piece (variable destination)
     wall_height is the index of the lowest empty row so we start from there and look downwards */
  piece_height = oriented_piece->height;
  piece_width = oriented_piece->width;
  destination = wall_height;
  
  collision = 0;
  while (destination >= 0 && !collision) { /* descend while no collision */
  
    /* detect collisions on each row occupied by the piece */
    current_row = destination;

    for (i = 0; i < piece_height && !collision; i++, current_row++) {

      /* debug info
      if (destination + i >= board->height || destination < 0) {
	printf("on déborde ! destination = %d, i = %d, wall_height = %d, piece_height = %d\n",
	       destination, i, wall_height, piece_height);
	board_print(stdout, board);
      }
      */

      collision = board_rows[current_row] & (piece_bricks[i] >> column);
    }
    if (!collision) {
      destination--;
    }
  }
  destination++;
  /* now destination is the index of the row where the bottom part of the piece is put */
  
  destination_top = destination + piece_height;

  /* update wall_height */
  wall_height = MAX(wall_height, destination_top);
  
  /* update the board */
  for (i = 0; i < piece_height; i++) {
    board_rows[destination + i] |= piece_bricks[i] >> column;
  }

  /* remove full rows */

  if (destination_top <= board->height || board->allow_lines_after_overflow) {

    i = 0;
    i_stop = piece_height;
    while (i < i_stop) {
      current_row = destination + i;

      /* check whether the row is full */
      if (board_rows[current_row] == full_row) {
	/* move the upper rows downwards */
	j = current_row;
	while (j < wall_height - 1 && board_rows[j] != empty_row) {
	  board_rows[j] = board_rows[j+1];
	  j++;
	}
      
	/* clear the last row */
	board_rows[j] = empty_row;
	wall_height--;

	/* update last_move_info */
	if (last_move_info != NULL) {
	  /* 	printf("updating eliminated bricks: row %d is full\n", current_row); */
	  /* 	printf("corresponds to row %d in the piece\n", i + removed_lines); */
	  /* 	printf("nb cells: %d\n", oriented_piece->nb_full_cells_on_rows[i + removed_lines]); */
	  /* 	print_piece_orientation(stdout, oriented_piece); */
	  last_move_info->eliminated_bricks_in_last_piece += oriented_piece->nb_full_cells_on_rows[i + removed_lines];
	}

	removed_lines++;
	i_stop--;
      }
      else {
	i++;
      }
    }
  }

  /* update last_move_info */
  if (last_move_info != NULL) {
    last_move_info->removed_lines = removed_lines;
    last_move_info->landing_height_bottom = destination;
    last_move_info->column = column;
    last_move_info->orientation = orientation;
    last_move_info->oriented_piece = oriented_piece;
  }

  board->wall_height = wall_height;
  
  return removed_lines;
}


/**
 * @brief Drops a new piece onto the wall (implementation that is close(r) to the original Tetris setting)
 * 
 * This function is called by game_drop_piece().
 * The move is supposed to be authorized, otherwise the result of this function is undetermined.
 *
 * @param board the board
 * @param oriented_piece the oriented piece to add
 * @param orientation the piece orientation (used to update \c last_move_info)
 * @param column where to drop the piece (the left column of the piece is considered),
 * the first column is \c 1
 * @param last_move_info if not \c NULL, some information about this move will be stored here
 * (useful for some of the features)
 * @param cancellable true to allow the move to be cancelled later
 * @return the number of removed lines
 * @see board_cancel_last_move()
 */
int board_drop_piece_rlc(Board *board, Piece *pieces, int piece_index, int desired_orientation, int desired_column,
		     LastMoveInfo *last_move_info, int cancellable) {
  int i, i_stop, j;
  int destination;              /* index of the row where the bottom part of the piece is put */
  int destination_top;          /* 1 + (index of the highest row occupied by the piece once it is put) */           
  uint16_t *board_rows;
  uint16_t *piece_bricks;
  uint16_t empty_row, full_row;
  int collision;
  int piece_height, piece_width;
  int current_row;
  int removed_lines;
  int wall_height;
  int column, orientation,previous_column,previous_orientation,previous_destination;
  Piece *piece;
  PieceOrientation *oriented_piece;
  int first_action_done;

  int or_todo, col_todo;
  
  /* In the rlc implementation, the pieces slightly move horizontally/vertically when they are rotated. The following offsets allow to deal with this. */
  static const int offsets[7][4][2]={ {{2,-2}, {0,0},  /* I */ {0,0},{0,0}},
				      {{0,0}, /* O */ {0,0},{0,0},{0,0}},
				      {{1,0},  {0,1}, {0,0}, {0,0}}, /* T */
				      {{0,-1}, {0,0},  /* Z */ {0,0},{0,0}},
				      {{0,-1}, {0,0},  /* S */ {0,0},{0,0}},
				      {{0,1},  {0,0}, {0,0}, {1,0}, },  /* J */  
				      {{0,-1}, {1,-1}, {0,0}, {0,-1}} /* L */				      		            };
				
  piece=&pieces[piece_index];

  board_rows = board->rows;
  empty_row = board->empty_row;
  full_row = board->full_row;
  
  wall_height = board->wall_height;
  removed_lines = 0;

  /* backup the board if necessary */
  if (cancellable) {
    MEMCPY(board->previous_rows, board_rows, uint16_t, board->extended_height);
    board->previous_wall_height = wall_height;
  }

  /* initialize last_move_info */
  last_move_info->eliminated_bricks_in_last_piece = 0;   
  last_move_info->nb_steps = 2;  
  
  /* we will search the lowest available row for the piece (variable destination)
     we start from the top of the board, progressively move the piece until it collides or touches the ground */

  /* we begin by setting the initial configuration */  
  orientation = piece->nb_orientations/2; /* the orientation that is presented: 1 or->0, 2 or->1, 4 or->2 */  
  /* set the information concerning the piece */
  oriented_piece = &piece->orientations[orientation];
  piece_bricks = oriented_piece->bricks;
  piece_height = oriented_piece->height;
  piece_width = oriented_piece->width;
  column = (board->width)/2; /* at the beginning: put the piece at the center */
  if (piece_index==0) { /* special treatment for piece I */
    column--; 
  }

  destination = board->height-piece_height; /* height coordinate of the piece at the beginning of the fall (the top of the piece touches the top of the board */  
    
  collision = 0;

  first_action_done = 0;
   
  or_todo=desired_orientation-orientation;
  col_todo=(desired_column-offsets[piece_index][desired_orientation][0])-(column-offsets[piece_index][orientation][0]);

  
  /*printf ("*** Put piece %i *** desired or/col=(%i,%i), current y/or/col=(%i,%i)(%i)\n", piece_index, desired_orientation, desired_column, orientation, column,destination); */

  while (destination >= 0 && !collision) { /* descend while no collision */          

    /* detect collisions on each row occupied by the piece */
    current_row = destination;
    for (i = 0; i < piece_height && !collision; i++, current_row++) {
      collision = board_rows[current_row] & (piece_bricks[i] >> column);      
    }
    if (!collision) {

      /* debug: show the piece on the board */
      /* 1) add the piece to the board */
/*       printf ("desired or/col=(%i,%i), current y/or/col=(%i, %i,%i)\n",desired_orientation, desired_column,destination, orientation, column);  */
/*       printf("Still need %i rotations and %i translations\n",or_todo,col_todo);*/

/*       for (i = 0; i < piece_height; i++) { */
/* 	board_rows[destination + i] |= piece_bricks[i] >> column;  */
/*       } */
/*       board_print(stdout, board);  */
/*       getchar();        */
/*       for (i = 0; i < piece_height; i++) { */
/* 	board_rows[destination + i] &= ~(piece_bricks[i] >> column); */
/*       } */
      /* end debug */      

      /* possibly rotate/translate */
      previous_orientation=orientation;
      previous_column=column;
      previous_destination=destination;

      /* the following mimicks the behavior of the python agent */
      if (first_action_done==0) { /* first action is a translation... */
	if (col_todo<0) {
	  column--;
	  col_todo++;
	  last_move_info->nb_steps++; 
	} else if (col_todo>0) {
	  column++;
	  col_todo--;
	  last_move_info->nb_steps++; 
	} else if ((piece_index!=0) && (piece_index!=5)) { /* ...or a rotation if there is no translation and the piece is neither J nor I (or else nothing) */
	  if (or_todo<0) { 
	    orientation--;
	    or_todo++;
	    last_move_info->nb_steps++; 
	  } else if (or_todo>0) {
	    orientation++;
	    or_todo--;
	    last_move_info->nb_steps++; 
	  } 
	} else if (or_todo!=0) { /* if piece if J or I and need for rotation */
	  last_move_info->nb_steps++;	  
	}
      }
      else { /* subsequent actions put priority on rotations */
	if (or_todo<0) {
	  orientation--;
	  or_todo++;
	  last_move_info->nb_steps++; 
	} else if (or_todo>0) {
	  orientation++;
	  or_todo--;
	  last_move_info->nb_steps++; 
	} else if (col_todo<0) {
	  column--;
	  col_todo++;
	  last_move_info->nb_steps++; 
	} else if (col_todo>0) {
	  column++;
	  col_todo--;
	  last_move_info->nb_steps++; 
	}
      }
      first_action_done=1;

      if ((previous_orientation!=orientation) || (previous_column!=column)) { /* if we tried to move the piece: check that it was possible */

	if (previous_orientation!=orientation) { /* if we have rotated the piece */
	  /* find its real new position: */

	  column += offsets[piece_index][orientation][0]-offsets[piece_index][previous_orientation][0];
	  destination += offsets[piece_index][orientation][1]-offsets[piece_index][previous_orientation][1];
	  /* update the corresponding variables */
	  oriented_piece = &piece->orientations[orientation];
	  piece_bricks = oriented_piece->bricks;
	  piece_height = oriented_piece->height;
	  piece_width = oriented_piece->width;
	}
	
	/* check for collision */    	
	if (destination+piece_height > board->height) { /* collision with the top on a rotation */
	  /*collision = 1;*/
	  DIE("Colliding with the top!!! this should not happen!!!\n"); 
	} else if (destination<0) { /* this might happen if a piece goes down when rotating and is blocked from the top. In this case, we cancel the rotation. */
	  collision = 1;
	} else {
	  current_row = destination;
	  for (i = 0; (i<piece_height) && !collision; i++, current_row++) { /* collision with other pieces */
	    collision = board_rows[current_row] & (piece_bricks[i] >> column);      
	  }
	}
	
	if (collision) { /* if it was not possible, cancel the move */

	  column=previous_column; /* we do this for rotation (recall that rotations can change the column) or translation */	  
	  if (previous_orientation!=orientation) { /* if we have rotated the piece */
	    orientation=previous_orientation;
	    destination=previous_destination;
	    oriented_piece = &piece->orientations[orientation];
	    piece_bricks = oriented_piece->bricks;
	    piece_height = oriented_piece->height;
	    piece_width = oriented_piece->width;
	  }	  
	  collision=0; /* the move has been cancelled so everything's alright now */
	}
	
      }

      destination--; /* go down */

    }
  }
  destination++;
  /* now destination is the index of the row where the bottom part of the piece is put */
  
  destination_top = destination + piece_height;

  /* update wall_height */
  wall_height = MAX(wall_height, destination_top);
  
  /* update the board */
  for (i = 0; i < piece_height; i++) {
    board_rows[destination + i] |= piece_bricks[i] >> column;
  }
  /* debug: show final destination */
  /*  board_print(stdout, board); */
  /*  getchar(); */
  /* end debug */

  /* remove full rows */

  if (destination_top <= board->height || board->allow_lines_after_overflow) {

    i = 0;
    i_stop = piece_height;
    while (i < i_stop) {
      current_row = destination + i;

      /* check whether the row is full */
      if (board_rows[current_row] == full_row) {
	/* move the upper rows downwards */
	j = current_row;
	while (j < wall_height - 1 && board_rows[j] != empty_row) {
	  board_rows[j] = board_rows[j+1];
	  j++;
	}
      
	/* clear the last row */
	board_rows[j] = empty_row;
	wall_height--;

	/* update last_move_info */
	if (last_move_info != NULL) {
	  /* 	printf("updating eliminated bricks: row %d is full\n", current_row); */
	  /* 	printf("corresponds to row %d in the piece\n", i + removed_lines); */
	  /* 	printf("nb cells: %d\n", oriented_piece->nb_full_cells_on_rows[i + removed_lines]); */
	  /* 	print_piece_orientation(stdout, oriented_piece); */
	  last_move_info->eliminated_bricks_in_last_piece += oriented_piece->nb_full_cells_on_rows[i + removed_lines];
	}

	removed_lines++;
	i_stop--;
      }
      else {
	i++;
      }
    }
  }

  /* update last_move_info */
  if (last_move_info != NULL) {
    last_move_info->removed_lines = removed_lines;
    last_move_info->landing_height_bottom = destination;
    last_move_info->column = column;
    last_move_info->orientation = orientation;
    last_move_info->oriented_piece = oriented_piece;
  }

  board->wall_height = wall_height;
  
  return removed_lines;
}



/**
 * @brief Drops a new piece onto the wall (this function updates a fancy version of the board for X interface, with different colors for different pieces)
 * 
 * @param board the board
 * @param oriented_piece the oriented piece to add
 * @param orientation the piece orientation (used to update \c last_move_info)
 * @param column where to drop the piece (the left column of the piece is considered),
 * the first column is \c 1
 * @param last_move_info if not \c NULL, some information about this move will be stored here
 * (useful for some of the features)
 * @param cancellable true to allow the move to be cancelled later
 * @param fancy_board an array that contains the colors of the pieces
 * @return the number of removed lines
 * @see board_cancel_last_move()
 */
int board_drop_piece_fancy(Board *board, PieceOrientation *oriented_piece, int orientation, int column,
			   LastMoveInfo *last_move_info, int cancellable, int **fancy_board) {
  int i, i_stop, j, k;
  int destination;              /* index of the row where the bottom part of the piece is put */
  int destination_top;          /* 1 + (index of the highest row occupied by the piece once it is put) */           
  uint16_t *board_rows;
  uint16_t *piece_bricks;
  uint16_t empty_row, full_row;
  int collision;
  int piece_height, piece_width;
  int current_row;
  int removed_lines;
  int wall_height;

  board_rows = board->rows;
  empty_row = board->empty_row;
  full_row = board->full_row;
  piece_bricks = oriented_piece->bricks;
  wall_height = board->wall_height;
  removed_lines = 0;

  /* backup the board if necessary */
  if (cancellable) {
    MEMCPY(board->previous_rows, board_rows, uint16_t, board->extended_height);
    board->previous_wall_height = wall_height;
  }

  /* initialize last_move_info */
  if (last_move_info != NULL) {
    last_move_info->eliminated_bricks_in_last_piece = 0;
  }
  
  /* we will search the lowest available row for the piece (variable destination)
     wall_height is the index of the lowest empty row so we start from there and look downwards */
  piece_height = oriented_piece->height;
  piece_width = oriented_piece->width;
  destination = wall_height;
  
  collision = 0;
  while (destination >= 0 && !collision) { /* descend while no collision */
  
    /* detect collisions on each row occupied by the piece */
    current_row = destination;

    for (i = 0; i < piece_height && !collision; i++, current_row++) {

      /* debug info
      if (destination + i >= board->height || destination < 0) {
	printf("on déborde ! destination = %d, i = %d, wall_height = %d, piece_height = %d\n",
	       destination, i, wall_height, piece_height);
	board_print(stdout, board);
      }
      */

      collision = board_rows[current_row] & (piece_bricks[i] >> column);
    }
    if (!collision) {
      destination--;
    }
  }
  destination++;
  /* now destination is the index of the row where the bottom part of the piece is put */
  
  destination_top = destination + piece_height;

  /* update wall_height */
  wall_height = MAX(wall_height, destination_top);
  
  /* update the board and the fancy board */
  for (i = 0; i < piece_height; i++) {
    board_rows[destination + i] |= piece_bricks[i] >> column;

    /* update of the fancy board */
    for (j = 1; j <= board->width; j++) {
      if ((piece_bricks[i] >> column) & brick_masks[j]) {
	fancy_board[destination + i][j-1]=-1;
      }
    }
    
  }      

  /* remove full rows */

  if (destination_top <= board->height || board->allow_lines_after_overflow) {

    /* fancy board detect full rows */
    i = 0;
    i_stop = piece_height;
    while (i < i_stop) {
      current_row = destination + i;
      if (board_rows[current_row] == full_row) {
	for (k=0; k<board->width; k++) {
	  if (fancy_board[current_row][k]!=-1) {
	      fancy_board[current_row][k]=-2; /* to remove */
	  }
	}
      }
      i++;
    }

    i = 0;
    i_stop = piece_height;
    while (i < i_stop) {
      current_row = destination + i;

      /* check whether the row is full */
      if (board_rows[current_row] == full_row) {
	/* move the upper rows downwards */
	j = current_row;
	while (j < wall_height - 1 && board_rows[j] != empty_row) {
	  board_rows[j] = board_rows[j+1];	  
	  j++;
	}
      
	/* clear the last row */
	board_rows[j] = empty_row;
	wall_height--;

	/* update last_move_info */
	if (last_move_info != NULL) {
	  /* 	printf("updating eliminated bricks: row %d is full\n", current_row); */
	  /* 	printf("corresponds to row %d in the piece\n", i + removed_lines); */
	  /* 	printf("nb cells: %d\n", oriented_piece->nb_full_cells_on_rows[i + removed_lines]); */
	  /* 	print_piece_orientation(stdout, oriented_piece); */
	  last_move_info->eliminated_bricks_in_last_piece += oriented_piece->nb_full_cells_on_rows[i + removed_lines];
	}

	removed_lines++;
	i_stop--;
      }
      else {
	i++;
      }
    }
  }

  /* update last_move_info */
  if (last_move_info != NULL) {
    last_move_info->removed_lines = removed_lines;
    last_move_info->landing_height_bottom = destination;
    last_move_info->column = column;
    last_move_info->orientation = orientation;
    last_move_info->oriented_piece = oriented_piece;
  }

  board->wall_height = wall_height;
  
  return removed_lines;

}





/**
 * @brief Removes the last dropped piece and restores the board state.
 *
 * This is possible only if cancellable was set to 1 when you called
 * board_drop_piece().
 *
 * @param board the board
 * @see board_drop_piece()
 */
void board_cancel_last_move(Board *board) {
  uint16_t *tmp_rows;
  
  tmp_rows = board->rows;
  board->rows = board->previous_rows;
  board->previous_rows = tmp_rows;
  board->wall_height = board->previous_wall_height;
}

/**
 * @brief Makes the board empty to start a new game.
 * @param board the board to clear
 */
void board_reset(Board *board) {
  int i;
  for (i = 0; i < board->extended_height; i++) {
    board->rows[i] = board->empty_row;
  }
  board->wall_height = 0;
}

/**
 * @brief Returns the height of a column.
 *
 * This function returns a value computed by board_update_column_heights(),
 * so you board_update_column_heights() must have been called first.
 * Indeed, for performance reasons, the column heights are computed only
 * if you ask it explicitly by calling board_update_column_heights()).
 *
 * @param board a board
 * @param column the column to consider
 * @return the height of this column (zero for an empty column)
 * @see board_update_column_heights()
 */
int board_get_column_height(Board *board, int column) {

/*   printf("\n--------- board_get_column_height: column = %d, height = %d\n", column, board->column_heights[column]); */
/*   print_board(stdout, board); */
/*   getchar(); */

  return board->column_heights[column];
}

/**
 * @brief Computes the height of the columns.
 * 
 * Calculates the height of each column and stores it so that
 * board_get_column_height() can work.
 * When the board changes, this function has to be called if you
 * want board_get_column_height() to return the right value.
 * This function is quite costly because it loops on the columns
 * and the rows, whereas most of the functions don't need to 
 * loop on the columns.
 * If your algorithm doesn't need the height of each column,
 * you won't need to call board_get_column_height(), so
 * don't waste your time calling this function.
 *
 * @param board a board
 * @see board_get_column_height()
 */
void board_update_column_heights(Board *board) {
  int i, j, board_width, wall_height, *p_column_height;
  uint16_t *board_rows, column_mask;

  board_rows = board->rows;
  board_width = board->width;
  wall_height = board->wall_height - 1;
  p_column_height = &board->column_heights[1];

  /* for each column */
  for (j = 1; j <= board_width; j++, p_column_height++) {
    column_mask = brick_masks[j];
    *p_column_height = 0;
    for (i = wall_height; i >= 0; i--) {
      if (board_rows[i] & column_mask) {
	*p_column_height = i + 1;
	break;
      }
    }
  }
  
  /* debug */
/*   printf("\n-------- Updating column heights -----------\n"); */
/*   print_board(stdout, board); */
/*   for (j = 1; j <= board_width; j++) { */
/*     printf("column %d: height = %d\n", j, board->column_heights[j]); */
/*   } */
/*   getchar(); */

}

/**
 * @brief Prints the board into a file.
 * 
 * A human-readable view of the board is printed.
 *
 * @param out the file to write, e.g. stdout
 * @param board a board
 */
void board_print(FILE *out, Board *board) {
  int i, j;
  
  fprintf(out, "Wall height: %d\n", board->wall_height);

  for (i = board->wall_height; i >= board->height; i--) {

    fprintf(out, " ");

    for (j = 1; j <= board->width; j++) {
      if (board->rows[i] & brick_masks[j]) {
	fprintf(out, "X");
      }
      else {
	fprintf(out, " ");
      }
    }
    
    fprintf(out, "\n");
  }

  for (i = board->height - 1; i >= 0; i--) {

    fprintf(out, "|");

    for (j = 1; j <= board->width; j++) {
      if (board->rows[i] & brick_masks[j]) {
	fprintf(out, "X");
      }
      else {
	fprintf(out, ".");
      }
    }
    
    fprintf(out, "|\n");
  }
}
