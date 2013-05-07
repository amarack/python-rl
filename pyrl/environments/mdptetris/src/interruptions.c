#include "config.h"
#include "interruptions.h"

#ifdef HAVE_SIGNAL_H

#include <stdio.h>
#include <stdlib.h>
#define __USE_POSIX
#include <signal.h>

/**
 * Number of times the user has pressed Ctrl-C.
 */
static int nb_interruptions = 0;

/**
 * Default interruption handler, saved here so that
 * we can restore it.
 */
static struct sigaction old_sigaction;

/**
 * Function called when the user pressed Ctrl-C.
 */
static void interrupt_handler(int sig);

/**
 * Initializes the interruption handler.
 */
void initialize_interruptions(void) {
  struct sigaction new_sigaction;
  sigset_t sigset;

  /* set the interruption handler */
  new_sigaction.sa_handler = interrupt_handler;
  sigemptyset(&sigset);
  new_sigaction.sa_mask = sigset;
  new_sigaction.sa_flags = 0;
  sigaction(SIGINT, &new_sigaction, &old_sigaction);
}

/**
 * Restores the default interruption handler.
 */
void exit_interruptions(void) {
  sigaction(SIGINT, &old_sigaction, NULL);
}

/**
 * Returns whether the user pressed Ctrl-C a first time.
 */
int is_interrupted(void) {
  return nb_interruptions;
}

/**
 * Function called when the user pressed Ctrl-C.
 */
static void interrupt_handler(int sig) {
  switch (nb_interruptions) {

  case 0:
    printf("\nInterruption detected - Finishing the current iteration\nPress Ctrl-C again to exit now\n");
    break;

  case 1:
    printf("\n");
    exit(0);
    break;
  }

  nb_interruptions++;
}

#else

/* signal.h is not present: we disable the Ctrl-C system */

/**
 * @cond
 */

void initialize_interruptions(void) {

}

void exit_interruptions(void) {

}

int is_interrupted(void) {
  return 0;
}

/**
 * @endcond
 */

#endif
