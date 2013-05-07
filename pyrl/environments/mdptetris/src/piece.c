#include <stdlib.h>
#include <stdio.h>
#include "config.h"
#include "macros.h"
#include "piece.h"
#include "brick_masks.h"
#include "file_tools.h"

/*
 * Private function.
 */
static void piece_orientation_init(PieceOrientation *orientation, int width, int height);

/**
 * @brief Creates a set of pieces as described in a given file.
 *
 * The file describes the existing pieces of the game.
 * The standard Tetris pieces are defined in the file pieces4.dat.
 * The pieces are loaded from the file into the pointer given in parameter.
 *
 * @param file_name name of the file with the definition of the pieces
 * @param nb_pieces this pointer will receive the number of pieces
 * @param pieces address of a pointer to store the array of pieces loaded from the file
 */
void load_pieces(const char *file_name, int *nb_pieces, Piece **pieces) {
  FILE *file;

  int i, j, k, o;
  int nb_orientations, tmp, height, width;
  Piece *current_piece;
  PieceOrientation *current_orientation;

  char *line;
  int line_size;

  /* open the file */
  file = open_data_file(file_name, "r");
  if (!file) {
    DIE1("Could not read the pieces file: %s", file_name);
  }
  line_size = 255;
  MALLOCN(line, char, line_size);

  /* read the number of pieces */
  if (!readline_skipcomments(file, line, line_size)) {
    problem_reading_file(file_name, "number_of_pieces", "end_of_file");
  }
  else if (sscanf(line, "%d", nb_pieces) != 1) {
    problem_reading_file(file_name, "number_of_pieces", line);
  }

  MALLOCN(*pieces, Piece, *nb_pieces);

  for (i = 0; i < *nb_pieces; i++) {

    /* for each piece i, scan nb_orientations, width and height */
    if (!readline_skipcomments(file, line, line_size)) {
      problem_reading_file(file_name, "nb_orientations height width", "end_of_file");
    }
    else if (sscanf(line,"%d %d %d", &nb_orientations, &height, &width) != 3) {
      problem_reading_file(file_name, "nb_orientations height width", line);
    }

    /* get a pointer to the piece we will load */
    current_piece = &((*pieces)[i]);
    current_piece->nb_orientations = nb_orientations;

    MALLOCN(current_piece->orientations, PieceOrientation, nb_orientations);

    /* scan the piece and store its orientation 0 */
    current_orientation = &(current_piece->orientations[0]);
    piece_orientation_init(current_orientation, width, height);
    
    for (j = height - 1; j >= 0; j--) {
      readline_skipcomments(file, line, line_size);

      /* set bit 1 for each filled cell */
      for (k = 0; k < width; k++) {
	if (line[k] == 'X') {
	  current_orientation->bricks[j] |= brick_masks[k]; /* we set the bit k */
	  current_orientation->nb_full_cells_on_rows[j]++;  /* count the number of cells on that row */
	}
      }
    }
    
    /* rotate this piece to generate the other orientations, and store them */
    for (o = 1; o < current_piece->nb_orientations; o++) {
      
      /* height and width are swapped after each rotation */
      tmp = height;
      height = width;
      width = tmp;
      
      current_orientation = &(current_piece->orientations[o]);
      piece_orientation_init(current_orientation, width, height);
      
      for (j = 0; j < height; j++) {
	for (k = 0; k < width; k++) {
	  /* rotate the previous orientation */
	  if (current_piece->orientations[o-1].bricks[width-1-k] & brick_masks[j]) {
	    current_orientation->bricks[j] |= brick_masks[k];
	    current_orientation->nb_full_cells_on_rows[j]++;
	  }
	  /* without bit masks we would have something like:
	  current_orientation->bricks[j][k] = current_piece->orientations[o-1].bricks[k][height-1-j];
	  */
	}
      }
    } /* for each orientation*/

  } /* for each piece */
  
  FREE(line);
  fclose(file);
}

/**
 * @brief Initializes a piece orientation.
 *
 * Initializes an already allocated piece orientation with the given width and height.
 * The piece shape for this orientation is initialized with zeros.
 *
 * @param orientation pointer to the piece orientation to initialize
 * @param width number of columns occupied by the piece shape in this orientation
 * @param height number of rows occupied by the piece shape in this orientation
 */
static void piece_orientation_init(PieceOrientation *orientation, int width, int height) {
  orientation->width = width;
  orientation->height = height;

  /* allocate the memory for the bricks */
  CALLOC(orientation->bricks, uint16_t, height);
  CALLOC(orientation->nb_full_cells_on_rows, int, height);
}

/**
 * @brief Destroys a piece.
 * @param piece the piece to destroy
 */
void free_piece(Piece *piece) {
  int i;
  for (i = 0; i < piece->nb_orientations; i++) {
    free(piece->orientations[i].bricks);
    free(piece->orientations[i].nb_full_cells_on_rows);
  }
  free(piece->orientations);
}

/**
 * @brief Prints into a file a human-readable view of the first orientation of a piece.
 *
 * This is equivalent to <code>piece_print_orientation(out, &piece->orientations[0])</code>.
 *
 * @param out the file to write (e.g. stdout)
 * @param piece the piece to print
 * @see piece_print_orientation(), piece_print_orientations()
 */
void piece_print(FILE *out, Piece *piece) {
  piece_print_orientation(out, &piece->orientations[0]);
}

/**
 * @brief Prints into a file a human-readable view of a piece orientation.
 * @param out the file to write (e.g. stdout)
 * @param orientation the piece orientation to consider
 * @see piece_print(), piece_print_orientations()
 */
void piece_print_orientation(FILE *out, PieceOrientation *orientation) {
  int i, j;

  for (i = orientation->height - 1; i >= 0; i--) {
    fprintf(out, "      ");
    for (j = 0; j < orientation->width; j++) {
      if (orientation->bricks[i] & brick_masks[j]) {
	fprintf(out, "X");
      } else {
	fprintf(out, " ");
      }
    }
    fprintf(out, "\n");
  }
}

/**
 * @brief Prints into of file all the orientations of a piece.
 *
 * A human-readable view of the piece and all its orientations is printed.
 *
 * @param out the file to write (e.g. stdout)
 * @param piece the piece to print
 */
void piece_print_orientations(FILE *out, Piece *piece) {
  int i;
  PieceOrientation *orientation;

  fprintf(out, "    %d orientations\n", piece->nb_orientations);
  for (i = 0; i < piece->nb_orientations; i++) {
    orientation = &piece->orientations[i];
    fprintf(out, "    Orientation #%d: width = %d, height = %d\n", i, orientation->width, orientation->height);
    piece_print_orientation(out, orientation);
  }
}
