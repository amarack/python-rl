/**
 * @defgroup macro Macros
 * @ingroup api
 * @brief Definition of useful macros
 *
 * This header provides macros to display error messages, allocate and free some memory,
 * and some basic mathematical features.
 *
 * @{
 */
#ifndef MACROS_H
#define MACROS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * @name General macros
 * @{
 */
/**
 * @brief Converts the parameter as a <code>char*</code> constant string.
 * @param x the text to convert
 */
#define STRING(x) STRING2(x)
#define STRING2(x) #x
/**
 * @}
 */

/**
 * @name Warning and error messages
 *
 * These macros display warning messages or error messages, with the source file name and
 * the line. Each macro exists can be called with 0 to 3 parameters. For example, if your
 * error message contains 2 parameters, call ERROR2.
 *
 * @{
 */

/**
 * @brief Shows a warning message.
 * @param X warning message
 */
#define WARNING(X) fprintf(stderr, "WARNING [file %s] [line %d] %s\n", __FILE__ , __LINE__ , X)

/**
 * @brief Shows an error message.
 * @param X error message
 */
#define ERROR(X) fprintf(stderr, "ERROR [file %s] [line %d] %s\n", __FILE__ , __LINE__ , X)

/**
 * @brief Shows an error message and stops the program.
 * @param X error message
 */
#define DIE(X) do { ERROR(X); exit(1); } while (0)

#define WARNING1(X, p1) fprintf(stderr,"WARNING [file %s] [line %d] " X "\n", __FILE__ , __LINE__ , p1)
#define WARNING2(X, p1, p2) fprintf(stderr,"WARNING [file %s] [line %d] " X "\n", __FILE__ , __LINE__ , p1, p2)
#define WARNING3(X, p1, p2, p3) fprintf(stderr,"WARNING [file %s] [line %d] " X "\n", __FILE__ , __LINE__ , p1, p2, p3)
#define ERROR1(X, p1) fprintf(stderr,"ERROR [file %s] [line %d] " X "\n", __FILE__ , __LINE__ , p1)
#define ERROR2(X, p1, p2) fprintf(stderr,"ERROR [file %s] [line %d] " X "\n", __FILE__ , __LINE__ , p1, p2)
#define ERROR3(X, p1, p2, p3) fprintf(stderr,"ERROR [file %s] [line %d] " X "\n", __FILE__ , __LINE__ , p1, p2, p3)
#define DIE1(X, p1) do { ERROR1(X, p1); exit(1); } while (0)
#define DIE2(X, p1, p2) do { ERROR2(X, p1, p2); exit(1); } while (0)
#define DIE3(X, p1, p2, p3) do { ERROR3(X, p1, p2, p3); exit(1); } while (0)
/**
 * @}
 */

/**
 * @brief Checks an assertion, stopping the program on an error message if the assertion is violated.
 * @param cond the assertion to check
 */
#define ASSERT(cond) do { if (!(cond)) { DIE1("Assertion '%s' violated", #cond); } } while (0);
#define ASSERT1(cond, message) do { if (!(cond)) { DIE1("Assertion '%s' violated (" message ")", #cond); } } while (0)
#define ASSERT2(cond, message, p1) do { if (!(cond)) { DIE2("Assertion '%s' violated (" message ")", #cond, p1); } } while (0)

/**
 * @name Memory
 *
 * These macros replace the usual memory related functions such as malloc and its family.
 *
 * @{
 */

/**
 * @brief Frees a pointer and sets it to NULL.
 *
 * The program stops on an error message if the pointer is NULL.
 *
 * @param x pointer to free
 */
#define FREE(x) if (x != NULL) { free(x); x = NULL; } else { DIE("INVALID MEMORY FREE()\n"); }

/**
 * @brief Allocates some memory for a datatype.
 *
 * If there is not enough memory, the program stops on an error message.
 *
 * @param p pointer to where the memory will be allocated
 * @param type type of the data allocated
 */
#define MALLOC(p, type) if ((p = (type*) malloc(sizeof(type))) == NULL) { DIE("MEMORY FULL\n"); }

/**
 * @brief Allocates some memory for an array.
 *
 * If there is not enough memory, the program stops on an error message.
 *
 * @param p pointer to where the memory will be allocated
 * @param type type of the data allocated
 * @param nb number of elements to allocate
 */
#define MALLOCN(p, type, nb) if ((p = (type*) malloc(nb * sizeof(type))) == NULL) { DIE("MEMORY FULL\n"); }

/**
 * @brief Allocates some memory for an array and initializes it to zero.
 *
 * This is the same as \ref MALLOCN except that the memory is initialized to zero.
 *
 * @param p pointer to where the memory will be allocated
 * @param type type of the data allocated
 * @param nb number of elements to allocate
 */
#define CALLOC(p, type, nb) if ((p = (type*) calloc(nb, sizeof(type))) == NULL) { DIE("MEMORY FULL\n"); }

/**
 * @brief Reallocates some memory for an array.
 *
 * If there is not enough memory, the program stops on an error message. * 
 *
 * @param p pointer to where the memory is currently allocated
 * @param type type of the data allocated
 * @param nb number of elements to reallocate
 */
#define REALLOC(p, type, nb) if ((p = (type*) realloc(p, nb * sizeof(type))) == NULL) { DIE("MEMORY FULL\n"); }

/**
 * @brief Copies some memory.
 * @param dst destination pointer
 * @param src source pointer
 * @param type type of the data copied
 * @param nb number of elements to copy
 */
#define MEMCPY(dst, src, type, nb) memcpy(dst, src, nb * sizeof(type))

/**
 * @}
 */

/**
 * @name Reading and writing
 *
 * @{
 */
#define SCANF1(format, p1) if (scanf(format, p1) != 1) { DIE("scanf failed"); }
#define SCANF2(format, p1, p2) if (scanf(format, p1, p2) != 2) { DIE("scanf failed"); }
#define SCANF3(format, p1, p2, p3) if (scanf(format, p1, p2, p3) != 3) { DIE("scanf failed"); }
#define SCANF(format, p1) SCANF1(format, p1)

#define FSCANF1(file, format, p1) if (fscanf(file, format, p1) != 1) { DIE("fscanf failed"); }
#define FSCANF2(file, format, p1, p2) if (fscanf(file, format, p1, p2) != 2) { DIE("fscanf failed"); }
#define FSCANF3(file, format, p1, p2, p3) if (fscanf(file, format, p1, p2, p3) != 3) { DIE("fscanf failed"); }
#define FSCANF(file, format, p1) FSCANF1(file, format, p1)

#define FREAD(ptr, size, count, file) if (fread(ptr, size, count, file) != count) { DIE("fread failed"); }
#define FWRITE(ptr, size, count, file) if (fwrite(ptr, size, count, file) != count) { DIE("fwrite failed"); }

/**
 * @}
 */

/**
 * @name Mathematics
 *
 * These macros provide simple mathematical operations.
 *
 * @{
 */
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define TETRIS_INFINITE (1e100)

#define TETRIS_SMALL_NON_ZERO 1e-30 /* or something else small */
#define DOUBLE_EQUAL(X,Y) ( fabs((X) - (Y)) < TETRIS_SMALL_NON_ZERO ) /* X == Y */
#define DOUBLE_GREATER_THAN(X,Y) ( (X) - (Y) > TETRIS_SMALL_NON_ZERO ) /* X > Y */
#define DOUBLE_LOWER_THAN(X,Y) ( (X) - (Y) < -TETRIS_SMALL_NON_ZERO ) /* X < Y */

/**
 * @}
 */

#endif

/**
 * @}
 */
