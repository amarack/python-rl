#ifndef RANDOM_H
#define RANDOM_H

/**
 * Initializes the GSL random number generator.
 */
void initialize_random_generator(unsigned int seed);

/**
 * Frees the random number generator.
 */
void exit_random_generator();

/**
 * Returns an integer number in [a,b[.
 */
int random_uniform(int a, int b);

#endif
