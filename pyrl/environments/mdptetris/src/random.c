#include <time.h>
#include <stdlib.h>
#include "config.h"
#include "random.h"

/**
 * Initializes the GSL random number generator with a specified seed.
 */
void initialize_random_generator(unsigned int seed) {
  srand(seed);
}

/**
 * Returns an integer number in [a,b[.
 */
int random_uniform(int a, int b) {
  return (rand() % (b - a)) - a;
}
