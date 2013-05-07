/**
 * @defgroup hashtable Hashtables
 * @ingroup api
 * @brief A hashtable implementation
 *
 * This module a hashtable implementation. A hashtable associates
 * a void* object to a string.
 *
 * @{
 */
#ifndef HASHTABLE_H
#define HASHTABLE_H

typedef struct Hashtable Hashtable;

/**
 * @name Hashtable creation and destruction
 * @{
 */
Hashtable* hashtable_new(int table_size, void (*free_function)(void *element));
void hashtable_free(Hashtable *hashtable);
/**
 * @}
 */

/**
 * @name Accessing the elements
 * @{
 */
void* hashtable_get(Hashtable *hashtable, const char *key);
void hashtable_add(Hashtable *hashtable, const char *key, void *data);
int hashtable_get_length(Hashtable *hashtable);
int hashtable_contains(Hashtable *hashtable, const char *key);
void hashtable_foreach(Hashtable *hashtable, void (*function)(const char *key, void *data));
/**
 * @}
 */

/**
 * @name Removing elements
 * @{
 */
void hashtable_remove(Hashtable *hashtable, const char *key);
void hashtable_clear(Hashtable *hashtable);
void hashtable_prune(Hashtable *hashtable, int (*should_remove)(const char *key, void *data));
/**
 * @}
 */

/**
 * @name Displaying (for debug)
 * @{
 */
void hashtable_print(Hashtable *hashtable);
/**
 * @}
 */

#endif
/**
 * @}
 */
