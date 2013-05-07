/**
 * @defgroup interruptions Interruptions
 * @ingroup api
 * @brief Ctrl-C signal management
 *
 * This module handles the Ctrl-C signal.
 * When the user presses Ctrl-C, a message is displayed.
 * Then your algorithm should finish the current iteration and stop.
 * If the user presses Ctrl-C again, the program exits.
 *
 * Note that this module uses the POSIX signals mechanism, and requires
 * the \c signal.h header. If \c signal.h is not detected by the \c configure
 * script, then the mechanism is disabled: the functions of this module
 * do nothing, and when the user presses Ctrl-C, the program is stopped.
 *
 * @{
 */
#ifndef INTERRUPTIONS_H
#define INTERRUPTIONS_H

/**
 * @brief Initializes the interruption handler.
 *
 * Call this function if you want to use the Ctrl-C system.
 * This function changes the handler of the SIGINT signal.
 * After this function is called, if SIGINT is received (i.e.
 * the user has pressed Ctrl-C), a message "Finishing the
 * current iteration" is displayed. Then the function
 * is_interrupted() returns \c 1 and your algorithm should
 * finish its current iteration and stop.
 * If the SIGINT signal is received a second time (i.e. after the message
 * was displayed but before the current iteration of your algorithm
 * is finished), then the program stops.
 *
 * @see exit_interruptions()
 *
 */
void initialize_interruptions(void);

/**
 * @brief Restores the default interruption handler.
 *
 * Call this function to cancel the behavior created by
 * initialize_interruptions().
 *
 * @see initialize_interruptions()
 */
void exit_interruptions(void);

/**
 * @brief Returns whether the user pressed Ctrl-C a first time.
 *
 * initialize_interruptions() should have been called before.
 * Your algorithm has to call this function to know when the user wants to
 * stop. As soon as this function returns 1, your algorithm should finish
 * its current iteration, save some data if necessary and then stop.
 *
 * @return 1 if the user pressed Ctrl-C once, and 0 otherwise.
 */
int is_interrupted(void);

#endif
/**
 * @}
 */
