/**
 * @defgroup file_tools File tools
 * @ingroup api
 * @brief Basic file parsing functions
 *
 * This module provides some utility functions to analyse a file.
 *
 * @{
 */
#ifndef FILE_TOOLS_H
#define FILE_TOOLS_H

#include <stdio.h>

/**
 * @brief Maximum number or characters allowed in a file name.
 */
#define MAX_FILE_NAME 256

/**
 * @brief Constant string containing the name of the MdpTetris data directory.
 */
#define DATADIR STRING(DATADIR_)


FILE *open_data_file(const char *file_name, const char *fopen_mode);
int readline_skipcomments(FILE *f, char *line, int line_size);
void problem_reading_file(const char *file_name, const char *expected, const char* readed);

#endif

/**
 * @}
 */
