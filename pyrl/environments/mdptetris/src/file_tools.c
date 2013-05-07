#include "config.h"
#include "file_tools.h"
#include "macros.h"

/**
 * @brief Opens a file in the current directory or in the MdpTetris data directory.
 *
 * The current directory is used for the user-defined data files. The data directory
 * contains data files installed with the program.
 *
 * First, the function tries to open the file in the current directory. If this file
 * doesn't exist, then the function tries to open it in the data directory of MdpTetris
 * (e.g. \c /usr/local/share/mdptetris).
 *
 * You should use this function instead of \c fopen to open any data file (provided
 * with the application of user-defined). You don't have to care about the directories.
 *
 * Note that the data directory contains read-only files installed with MdpTetris,
 * so this function will not try to open a file in the data directory if the fopen mode
 * is not \c "r".
 *
 * @param file_name name of the file to open
 * @param fopen_mode mode to give to the \c fopen call
 * @return the file, or NULL if it couldn't be open.
 */
FILE *open_data_file(const char *file_name, const char *fopen_mode) {

  FILE *f;
  char file_name_in_datadir[MAX_FILE_NAME];

  /* first, open the file in the current directory */
  f = fopen(file_name, fopen_mode);

  if (f == NULL && fopen_mode[0] == 'r') {
    /* open the file in the data directory */
    sprintf(file_name_in_datadir, "%s/%s", DATADIR, file_name);
    f = fopen(file_name_in_datadir, fopen_mode);
  }

  return f;
}

/**
 * @brief Reads the next non-comment line of a file.
 *
 * A line is considered as a comment if the first character is '#'.
 *
 * @param f the file to read
 * @param line pointer to store the characters that will be read
 * @param line_size maximum number of characters to read on the line
 * @return 1 if the a line was successfuly read, 0 if the end of the file
 * was reached.
 */
int readline_skipcomments(FILE *f, char *line, int line_size) {

  do {
    if (fgets(line, line_size, f) == NULL) {
      return 0;
    }

  } while (line[0] == '#');  /* skip comments */

  return 1;
}

/**
 * @brief Displays a message explaining an error occured when parsing a file
 * and exits the program.
 * @param file_name name of the file
 * @param expected a string describing what was expected
 * @param readed a string describing what was read instead of what was expected
 */
void problem_reading_file(const char *file_name, const char *expected, const char* readed) {
  DIE3("Problem reading file '%s': expected '%s' and readed '%s'\n", file_name, expected, readed);
}
