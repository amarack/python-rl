/**
 * Main file for Tetris.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "game.h"
#include "random.h"
#include "macros.h"
#include "file_tools.h"

static void play_interactive_game(int tetris_implementation, int width, int height, int allow_lines_after_overflow, const char *piece_file_name, int *piece_sequence);
static void choose_game_configuration(int *width, int *height, int *allow_lines_after_overflow, char *piece_file_name);
static int *choose_piece_sequence(void);
static void choose_random_generator_seed(void);

/**
 * Lets the user play a game.
 */
static void play_interactive_game(int tetris_implementation, int width, int height, int allow_lines_after_overflow, const char *piece_file_name, int *piece_sequence) {
  Game *game;
  Action action;
 
  /*game = new_standard_game();*/
  game = new_game(tetris_implementation, width, height, allow_lines_after_overflow, piece_file_name, piece_sequence);

  do {
    game_print(stdout, game);

    printf("Piece orientation (0-%d): ", game_get_nb_possible_orientations(game) - 1);
    SCANF1("%d", &action.orientation);
    printf("Column (1-%d): ", game_get_nb_possible_columns(game, action.orientation));
    SCANF1("%d", &action.column);

    game_drop_piece(game, &action, 0);
  }
  while (!game->game_over);

  game_print(stdout, game);

  free_game(game);
}

/**
 * Lets the user choose the game configuration: the pieces and the board dimensions.
 */
static void choose_game_configuration(int *width, int *height, int *allow_lines_after_overflow, char *piece_file_name) {
  int user_width, user_height, user_overflow, user_pieces;

  /* width */
  printf("  Board width : ");
  SCANF("%d", &user_width);

  while (user_width <= 2 || user_width > 14) {
    printf("  The board width must be in [2..14].\n");
    SCANF("%d", &user_width);
  }

  /* height */
  printf("  Board height : ");
  SCANF("%d", &user_height);

  while (user_height <= 1) {
    printf("  The board height must be greater than 1.\n");
    SCANF("%d", &user_height);
  }

  /* overflow */
  user_overflow = 0;
  while (user_overflow < 1 || user_overflow > 2) {
    printf("  Allow lines to be completed when the piece overflows?\n");
    printf("  1. No\n");
    printf("  2. Yes\n");
    printf("  Your choice: ");
    SCANF("%d", &user_overflow);
  }

  /* piece set */
  user_pieces = 0;
  while (user_pieces < 1 || user_pieces > 2) {
    printf("  Select a set of pieces:\n");
    printf("  1. Standard Tetris pieces (tetraminos)\n");
    printf("  2. Melax's reduced set of pieces\n");
    printf("  Your choice: ");
    SCANF("%d", &user_pieces);
  }

  switch (user_pieces) {
  case 1:
    strcpy(piece_file_name, "pieces4.dat");
    break;
  case 2:
    strcpy(piece_file_name, "pieces_melax.dat");
    break;
  }
  
  *width = user_width;
  *height = user_height;
  *allow_lines_after_overflow = user_overflow - 1;
}


/**
 * Lets the user choose a sequence of pieces. The pieces will
 * be generated from this sequence instead of being randomly chosen.
 * Returns an array of piece indexes. The last element of the array is -1.
 * Returns NULL if the user wants the pieces to be generated randomly.
 */
static int *choose_piece_sequence(void) {
  int choice, i, sequence_length;
  int *piece_sequence = NULL;
  char *sequence_file_name;
  FILE *sequence_file;

  printf("  1. Random pieces\n");
  printf("  2. S/Z tetraminos alternance\n");
  printf("  3. Set a custom sequence\n");
  printf("  4. Load a sequence from a file\n");
  printf("  Your choice: ");

  SCANF("%d", &choice);

  switch (choice) {
  case 1:
    piece_sequence = NULL;
    break;

  case 2:
    MALLOCN(piece_sequence, int, 3);
    piece_sequence[0] = 3;
    piece_sequence[1] = 4;
    piece_sequence[2] = -1;
    break;

  case 3:
    printf("  Number of pieces in a sequence: ");
    SCANF("%d", &sequence_length);
    MALLOCN(piece_sequence, int, sequence_length + 1);
    for (i = 0; i < sequence_length; i++) {
      printf("  Piece %d: ", i);
      SCANF("%d", &piece_sequence[i]);
    }
    piece_sequence[i] = -1; /* to indicate the end of the array */    
    break;

  case 4:
    MALLOCN(sequence_file_name, char, MAX_FILE_NAME);
    printf("  Name of file containing the sequence: ");
    SCANF("%s", sequence_file_name);

    sequence_file = open_data_file(sequence_file_name, "r");
    FREE(sequence_file_name);
    if (sequence_file == NULL) {
      DIE1("Cannot read the feature file '%s'", sequence_file_name);
    }
    FSCANF(sequence_file,"%d", &sequence_length); /* the first line contains the number of pieces in the file */
    MALLOCN(piece_sequence, int, sequence_length + 1);
    for (i = 0; i < sequence_length; i++) {
      FSCANF(sequence_file,"%d", &(piece_sequence[i]));   
    }    
    piece_sequence[i] = -1; /* to indicate the end of the array */    


    /*    fclose(sequence_file); */ /* BUG?*/

    
    break;
  }
  
  return piece_sequence;
}

/**
 * Lets the user choose the seed of the random number generator
 * and reinitializes the generator with this seed.
 */
static void choose_random_generator_seed(void) {
  unsigned int seed;

  printf("  Seed: ");
  SCANF("%ud", &seed);

  initialize_random_generator(seed);
}

/**
 * Main function.
 * Usage: ./tetris
 */
int main(int argc, char **argv) {

  int tetris_implementation;
  int choice;
  int width, height, allow_lines_after_overflow;
  char piece_file_name[256];
  int *piece_sequence;
  
  initialize_random_generator(time(NULL));

  choice = 0;

  /* default configuration */
  tetris_implementation = 0; /* Simplified */
  width = 10;
  height = 20;
  allow_lines_after_overflow = 0;
  strcpy(piece_file_name, "pieces4.dat"); 
  piece_sequence = NULL; /* random pieces by default */

  while (choice != 9) {

    printf("--- Main menu ---\n");
    printf("0. Set Tetris implementation (to be done, default=simplified)\n");
    printf("1. Play a game\n");
    printf("2. Set the game configuration\n");
    printf("3. Set the sequence of pieces\n");
    printf("4. Set the seed of the random number generator\n");
    printf("5. Quit\n");
    printf("Your choice: ");
    SCANF("%d", &choice);
    while (getchar() != '\n');
    printf("\n");
    
    switch (choice) {
      
    case 1:
      printf("Play a game\n");
      play_interactive_game(tetris_implementation,width, height, allow_lines_after_overflow, piece_file_name, piece_sequence);
      break;
      
    case 2:
      printf("Game configuration\n");
      choose_game_configuration(&width, &height, &allow_lines_after_overflow, piece_file_name);
      printf("Configuration saved.\n");
      break;

    case 3:
      printf("Sequence of pieces\n");
      piece_sequence = choose_piece_sequence();
      printf("Configuration saved.\n");
      break;

    case 4:
      printf("Seed of the random number generator\n");
      choose_random_generator_seed();
      printf("Seed initialized.\n");
      break;
    

    }
    printf("\n");
  }

  if (piece_sequence != NULL) {
    FREE(piece_sequence);
  }

  return 0;
}
