/**
 * This module contains functions to parse the common parameters from the command line
 * or to ask them to the user.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "common_parameters.h"

static RewardFunctionID default_reward_function_id = REWARD_REMOVED_LINES;

/**
 * Sets the default reward function.
 * This reward function will be set if you call load_default_parameters()
 * (the batch version).
 * If you call ask_common_parameters (the interactive version), this will
 * be the default choice proposed to the user.
 *
 * If you don't specify a default reward function by calling this function,
 * the default reward function is set to REWARD_REMOVED_LINES.
 */
void set_default_reward_function(RewardFunctionID reward_function_id) {
  default_reward_function_id = reward_function_id;
}

/**
 * Loads the default parameters.
 */
void load_default_parameters(CommonParameters *parameters) {
  RewardFunctionID reward_function_id;

  reward_function_id = default_reward_function_id;
  parameters->board_width = 10;
  parameters->board_height = 20;
  parameters->tetris_implementation = 0;
  parameters->allow_lines_after_overflow = 0;
  strcpy(parameters->piece_file_name, "pieces4.dat");
  parameters->random_generator_seed = time(NULL);

  parameters->reward_description.reward_function_id = reward_function_id;
  parameters->reward_description.reward_function = all_reward_functions[reward_function_id];
}

/**
 * Asks all common parameters to the user.
 */
void ask_common_parameters(CommonParameters *parameters) {
  char *read;
  char line[MAX_LENGTH];
  int reward_function_id;

  /* type of implementation  */
  printf("Tetris implementation (default 0):\n"
	 "  0. Simplified\n"
	 "  1. RLC\n"
	 "Your choice: ");
  
  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] < '0' || line[0] > '1') {
    parameters->tetris_implementation=0;
  }
  else {
    parameters->tetris_implementation = line[0] - '0';
  }

  /* board width */
  printf("Board width (default 10): ");
  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] == '\n') {
    parameters->board_width = 10;
  }
  else {
    sscanf(line, "%d", &parameters->board_width);
  }

  /* board height */
  printf("Board height (default 20): ");
  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] == '\n') {
    parameters->board_height = 20;
  }
  else {
    sscanf(line, "%d", &parameters->board_height);
  }

  /* allow overflow? */
  printf("Allow to remove lines when the board overflows (y/n, default n): ");
  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] != 'y') {
    parameters->allow_lines_after_overflow = 0;
  }
  else {
    parameters->allow_lines_after_overflow = 1;
  }

  /* pieces */
  printf("File describing the pieces (default pieces4.dat): ");
  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] == '\n') {
    strcpy(parameters->piece_file_name, "pieces4.dat");
  }
  else {
    line[strlen(line)-1] = '\0'; 
    strcpy(parameters->piece_file_name, line);
  }

  /* random generator seed */
  printf("Seed of the random number generator (default auto): ");
  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] == '\n') {
    parameters->random_generator_seed = time(NULL);
  }
  else {
    sscanf(line, "%ud", &parameters->random_generator_seed);
  }

  /* reward function */
  printf("Reward function (default %d):\n"
	 "  0. No reward\n"
	 "  1. Number of lines removed in the last move\n"
	 "  2. 1 for each move\n"
	 "  3. 1 if one or more lines were removed in the last move\n"
	 "  4. 1,4,9,15 rewards\n"
	 "Your choice: ",
	 default_reward_function_id);

  read = fgets(line, MAX_LENGTH, stdin);
  if (read == NULL || line[0] < '0' || line[0] > '4') {
    reward_function_id = default_reward_function_id;
  }
  else {
    reward_function_id = line[0] - '0';
  }
  parameters->reward_description.reward_function_id = reward_function_id;
  parameters->reward_description.reward_function = all_reward_functions[reward_function_id];
}

/**
 * Parses a single common parameter.
 * Returns the number of arguments read in the array args to parse this parameter.
 * For example, if the parameter is -height 10, the function returns 2.
 * The function returns 0 if the parameter is unknown.
 *
 * Parameters recognized:
 *
 * -width n                                      board width (default 10)
 * -height n                                     board height (default 20)
 * -allow_lines_after_overflow                   allow to remove lines when the piece overflows (defaut: disabled)
 * -pieces file_name                             file describing the pieces (default pieces4.dat)
 * -seed n                                       seed of the random number generator
 * -reward [none | lines | 1 | at_least_1_line]  reward function: no reward, number of lines removed,
 *                                               1 for each move, or 1 when at least one line is removed
 *
 * @param parameters the parameters where the parsed data will be stored
 * @param nb_args number of arguments in the array args
 * @param args array of arguments from the command line (only the first one is parsed)
 * @param print_usage a function to print the usage of your program's command line if an error occurs
 * @return the number of elements read in the array args to parse the first one
 */
int parse_common_parameter(CommonParameters *parameters, int nb_args, char **args, void (*print_usage)(void)) {
  char reward_function_name[MAX_LENGTH];
  char tetris_implementation_name[MAX_LENGTH];
  RewardFunctionID reward_function_id = NO_REWARD;

  /* tetris implementation */
  if (!strcmp(args[0], "-tetris_implementation")) {
    parameters_assert(nb_args > 1, "Missing argument to parameter -tetris_implementation", print_usage);
    
    parameters_assert(sscanf(args[1], "%s", tetris_implementation_name) == 1,
		      "Incorrect argument for parameter -tetris_implementation", print_usage);
    
    if (!strcmp(tetris_implementation_name, "simplified")) {
      parameters->tetris_implementation=0;
    }
    else if (!strcmp(tetris_implementation_name, "rlc")) {
      parameters->tetris_implementation=1;
    }
  }

  /* board width */
  else if (!strcmp(args[0], "-width")) {
    parameters_assert(nb_args > 1, "Missing argument to parameter -width", print_usage);
    parameters_assert(sscanf(args[1], "%d", &parameters->board_width) == 1,
		      "Incorrect argument for parameter -width", print_usage);
  }
  
  /* board height */
  else if (!strcmp(args[0], "-height")) {
    parameters_assert(nb_args > 1, "Missing argument to parameter -height", print_usage);
    parameters_assert(sscanf(args[1], "%d", &parameters->board_height) == 1,
		      "Incorrect argument for parameter -height", print_usage);
  }

  /* allow overflow */
  else if (!strcmp(args[0], "-allow_lines_after_overflow")) {
    parameters->allow_lines_after_overflow = 1;
  }

  /* pieces */
  else if (!strcmp(args[0], "-pieces")) {
    parameters_assert(nb_args > 1, "Missing argument to parameter -pieces", print_usage);
    parameters_assert(sscanf(args[1], "%s", parameters->piece_file_name) == 1,
		      "Incorrect argument for parameter -pieces", print_usage);
  }
  
  /* random number generator seed */
  else if (!strcmp(args[0], "-seed")) {
    parameters_assert(nb_args > 1, "Missing argument to parameter -seed", print_usage);
    parameters_assert(sscanf(args[1], "%ud", &parameters->random_generator_seed) == 1,
		      "Incorrect argument for parameter -seed", print_usage);
  }

  /* reward */
  else if (!strcmp(args[0], "-reward")) {
    parameters_assert(nb_args > 1, "Missing argument to parameter -reward", print_usage);
    
    parameters_assert(sscanf(args[1], "%s", reward_function_name) == 1,
		      "Incorrect argument for parameter -reward", print_usage);
    
    if (!strcmp(reward_function_name, "none")) {
      reward_function_id = NO_REWARD;
    }
    else if (!strcmp(reward_function_name, "lines")) {
      reward_function_id = REWARD_REMOVED_LINES;
    }
    else if (!strcmp(reward_function_name, "1")) {
      reward_function_id = REWARD_ONE;
    }
    else if (!strcmp(reward_function_name, "at_least_1_line")) {
      reward_function_id = REWARD_AT_LEAST_ONE_LINE;
    }
    else if (!strcmp(reward_function_name, "tetris_are_better")) {
      reward_function_id = REWARD_TETRIS_ARE_BETTER;
    }
    else {
      parameters_assert(0, "Unknown reward function", print_usage);
    }
    parameters->reward_description.reward_function_id = reward_function_id;
    parameters->reward_description.reward_function = all_reward_functions[reward_function_id];
  }

  /* unknown parameter */
  else {
    return 0;
  }

  /* 2 arguments have been read */
  return 2;
}

/**
 * Checks an assertion. If the assertion fails, an error message is displayed, the usage
 * of the command line is displayed and the program exits.
 * @param assertion the value of the assertion checked
 * @param error_message error message to show if the assertion fails
 * @param print_usage a function to print the usage of your program's command line if the assertion fails.
 */
void parameters_assert(int assertion, const char *error_message, void (*print_usage)(void)) {
  if (!assertion) {
    fprintf(stderr, "%s\n", error_message);
    print_usage();
    exit(1);
  }
}

/**
 * Prints a message explaining the command line syntax for the common parameters.
 */
void common_parameters_print_usage(void) {
  fprintf(stderr,
	  "-tetris_implementation [simplified | rlc]     type of Tetris implementation\n"
	  "-width n                                      board width (default 10)\n"
	  "-height n                                     board height (default 20)\n"
	  "-allow_lines_after_overflow                   allow to remove lines when the piece overflows (defaut: disabled)\n"
	  );

  fprintf(stderr,
	  "-pieces file_name                             file describing the pieces (default pieces4.dat)\n"
	  "-reward [none | lines | 1 | at_least_1_line | tetris_are_better]  reward function: number of lines removed, 1 for each move, (1,4,9,15)\n"
          "                                              or 1 each time one or more lines are made (default: lines)\n"
	  "-seed n                                       seed of the random number generator (default: time(NULL))\n"
	  );
}
