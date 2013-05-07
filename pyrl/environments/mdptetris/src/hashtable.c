#include "config.h"
#include "hashtable.h"
#include "macros.h"
#include <string.h>

typedef struct HashtableElement HashtableElement;

/**
 * @brief The hashtable structure.
 */
struct Hashtable {
  
  /**
   * @name The table
   */
  HashtableElement **table;                /**< The elements. */
  int table_size;                          /**< Capacity of the table. */
  int nb_elements;                         /**< Number of elements in the hashtable. */

  /**
   * @name Free function
   */
  void (*free_function)(void *element);   /**< Function to call to free the data stored in an element. */
};

/**
 * @brief An element stored in the hashtable.
 */
struct HashtableElement {
  
  char *key;                               /**< The element key, terminated by '\\0'. */
  void *data;                              /**< The element itself. */
  HashtableElement *next;                  /**< The next element (when several elements have the same hashcode). */
};

/*
 * Private functions for the elements
 */
static HashtableElement* hashtable_element_new(const char *key, void *data);
static void hashtable_element_free(HashtableElement *element, void (*free_function)(void *element));
static unsigned int get_hashcode(const char *key);

/**
 * @brief Creates a new hashtable.
 * @param table_size capacity of the table
 * @param free_function a function to call to free each element when destroying the table
 * (can be NULL, then the elements will not be freed)
 * @return the hashtable created
 * @see hashtable_free()
 */
Hashtable* hashtable_new(int table_size, void (*free_function)(void *element)) {

  Hashtable *hashtable;

  MALLOC(hashtable, Hashtable);
  CALLOC(hashtable->table, HashtableElement*, table_size);
  hashtable->table_size = table_size;
  hashtable->nb_elements = 0;
  hashtable->free_function = free_function;

  return hashtable;
}

/**
 * @brief Destroys a hashtable.
 *
 * If a free function was specified when creating the hashtable, then this
 * function is called on each element.
 *
 * @param hashtable the hashtable to free
 * @see hashtable_new()
 */
void hashtable_free(Hashtable *hashtable) {

  /* free the elements */
  hashtable_clear(hashtable);

  /* free the structure */
  FREE(hashtable->table);
  FREE(hashtable);
}

/**
 * @brief Returns an element.
 *
 * If no element with the specified key is in the hashtable, NULL is returned.
 *
 * @param hashtable the hashtable
 * @param key key of the element
 * @return the element
 */
void* hashtable_get(Hashtable *hashtable, const char *key) {

  HashtableElement *element;
  unsigned int hashcode;

  hashcode = get_hashcode(key) % hashtable->table_size;
  element = hashtable->table[hashcode];

  while (element != NULL && strcmp(key, element->key)) {
    element = element->next;
  }

  if (element == NULL) {
    return NULL;
  }
  else {
    return element->data;
  }
}

/**
 * @brief Returns whether the hashtable contains an element.
 * @param hashtable the hashtable
 * @param key key of the element
 * @return 1 if the hashtable contains this element, 0 otherwise
 */
int hashtable_contains(Hashtable *hashtable, const char *key) {

  HashtableElement *element;
  unsigned int hashcode;

  hashcode = get_hashcode(key) % hashtable->table_size;
  element = hashtable->table[hashcode];

  while (element != NULL && strcmp(key, element->key)) {
    element = element->next;
  }

  return (element != NULL);
}

/**
 * @brief Adds an element into the hashtable.
 *
 * @param hashtable the hashtable
 * @param key key of the element to add
 * @param data the data to add
 */
void hashtable_add(Hashtable *hashtable, const char *key, void *data) {

  HashtableElement *element;
  unsigned int hashcode;

  hashcode = get_hashcode(key) % hashtable->table_size;
  element = hashtable_element_new(key, data);
  element->next = hashtable->table[hashcode];
  hashtable->table[hashcode] = element;

  hashtable->nb_elements++;
}

/**
 * @brief Removes an element from the hashtable.
 *
 * If the element is not found, an error message is displayed and the program stops.
 *
 * @param hashtable the hashtable
 * @param key key of the element to remove
 */
void hashtable_remove(Hashtable *hashtable, const char *key) {

  HashtableElement *previous, *current;
  unsigned int hashcode;

  hashcode = get_hashcode(key) % hashtable->table_size;

  previous = NULL;
  current = hashtable->table[hashcode];
  while (current != NULL && strcmp(key, current->key)) {
    previous = current;
    current = current->next;
  }

  if (current == NULL) {
    DIE1("hashtable_remove(): no element with key '%s' in the hashtable", key);
  }

  if (previous == NULL) {
    /* it is the first element */
    hashtable->table[hashcode] = current->next;
  }
  else {
    previous->next = current->next;
  }

  hashtable_element_free(current, hashtable->free_function);
  hashtable->nb_elements--;
}

/**
 * @brief Returns the number of elements of a hashtable.
 * @param hashtable the hashtable
 * @return the number of elements in the hashtable
 */
int hashtable_get_length(Hashtable *hashtable) {
  return hashtable->nb_elements;
}

/**
 * @brief Removes all elements of a hashtable.
 */
void hashtable_clear(Hashtable *hashtable) {

  int i;
  HashtableElement *current, *next;

  for (i = 0; i < hashtable->table_size; i++) {
    
    current = hashtable->table[i];
    while (current != NULL) {
      next = current->next;
      hashtable_element_free(current, hashtable->free_function);
      current = next;
    }
    hashtable->table[i] = NULL;
  }
  
  hashtable->nb_elements = 0;
}

/**
 * @brief Executes a function on each element in the hashtable.
 *
 * Your function should not add or remove elements in the hashtable.
 *
 * @param hashtable the hashtable
 * @param function the function to execute on each element
 */
void hashtable_foreach(Hashtable *hashtable, void (*function)(const char *key, void *data)) {
  
  int i;
  HashtableElement *current, *next;

  for (i = 0; i < hashtable->table_size; i++) {
    
    current = hashtable->table[i];
    while (current != NULL) {
      next = current->next;
      function(current->key, current->data);
      current = next;
    }
  }
}

/**
 * @brief Removes from the hashtable all elements that verify a certain condition.
 * @param hashtable the hashtable
 * @param should_remove a boolean function that takes an element as parameter
 * and returns 1 if this element has to be removed
 */
void hashtable_prune(Hashtable *hashtable, int (*should_remove)(const char *key, void *data)) {

  int i;
  HashtableElement *previous, *current, *next;

  for (i = 0; i < hashtable->table_size; i++) {
    
    previous = NULL;
    current = hashtable->table[i];
    while (current != NULL) {

      if (should_remove(current->key, current->data)) {
	/* we have to remove current */

	if (previous == NULL) {
	  next = current->next;
	  hashtable_element_free(current, hashtable->free_function);
	  hashtable->table[i] = current = next;
	}
	else {
	  previous->next = current->next;
	  hashtable_element_free(current, hashtable->free_function);
	  current = previous->next;
	}

	hashtable->nb_elements--;
      }
      else {
	/* current is not removed */
	previous = current;
	current = current->next;
      }
    }
  }
}

/**
 * @brief Prints the structure of a hashtable (for debugging purposes only).
 * @param hashtable the hashtable to print
 */
void hashtable_print(Hashtable *hashtable) {

  int i;
  HashtableElement *current;

  for (i = 0; i < hashtable->table_size; i++) {
    
    printf("%d:\n", i);
    current = hashtable->table[i];
    while (current != NULL) {
      printf("  '%s' -> data\n", current->key);
      current = current->next;
    }
  }
}

/**
 * @brief Creates a hashtable element.
 * @param key key of the element to create
 * @param data the data of this element
 */
static HashtableElement* hashtable_element_new(const char *key, void *data) {

  HashtableElement *element;
  MALLOC(element, HashtableElement);

  MALLOCN(element->key, char, strlen(key) + 1);
  strcpy(element->key, key);

  element->data = data;
  element->next = NULL;

  return element;
}

/**
 * @brief Destroys a hashtable element.
 *
 * The key is destroyed, and the data is destroyed if \a free_function is not NULL.
 *
 * @param element the element to destroy
 * @param free_function a free function to destroy the data (can be NULL)
 */
static void hashtable_element_free(HashtableElement *element, void (*free_function)(void *element)) {

  if (free_function != NULL) {
    free_function(element->data);
  }
  free(element->key);
  FREE(element);
}

/**
 * @brief Returns a hashcode of a string.
 *
 * This function implements the djb2 algorithm (see for example
 * http://www.cse.yorku.ca/~oz/hash.html).
 *
 * @param key the key to hash
 * @return a hashcode
 */
static unsigned int get_hashcode(const char *key) {

  /* fast implementation of the djb2 algorithm */
  int c;
  unsigned int hash = 5381;
  
  while ((c = *key++)) {
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  }
  
  return hash;
}
