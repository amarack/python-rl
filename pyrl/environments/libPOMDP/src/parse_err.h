/* parse_err.h 

  *****
  Copyright 1994-1997, Brown University
  Copyright 1998, 1999, Anthony R. Cassandra

                           All Rights Reserved
                           
  Permission to use, copy, modify, and distribute this software and its
  documentation for any purpose other than its incorporation into a
  commercial product is hereby granted without fee, provided that the
  above copyright notice appear in all copies and that both that
  copyright notice and this permission notice appear in supporting
  documentation.
  
  ANTHONY CASSANDRA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
  PARTICULAR PURPOSE.  IN NO EVENT SHALL ANTHONY CASSANDRA BE LIABLE FOR
  ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  *****

	This module contains all the constant and type definitions
needed for the "pomdp_err.c" module. 
*/
#ifndef MDP_PARSE_ERR_H
#define MDP_PARSE_ERR_H 1

/***********************************************************************/
/****************************  CONSTANTS  ******************************/
/***********************************************************************/

#define DEBUG		1  /* set this constant to zero when not debugging */

/* Special values */
#define NO_LINE		-1	/* Value line number will take
				when a particular error message
				does not have an associated
				line number
				*/

#define ERR_META	'@'	/* Special character found in error messages
				that indicates an insertion point
				for another string.
				*/

/* Currently defined error messages */
#define NBR_ERRORS	29

#define ERROR_MSSG_0	"Illegal character @ in input."
#define ERROR_MSSG_1	"Bad value for discount factor. Valid range is 0 to 1."
#define ERROR_MSSG_2	"Bad number of states specified.  Must be > 0."
#define ERROR_MSSG_3	"Bad number of actions specified.  Must be > 0."
#define ERROR_MSSG_4    "Bad number of observations specified.  Must be > 0."
#define ERROR_MSSG_5    "State number is out of range."
#define ERROR_MSSG_6    "Action number is out of range."
#define ERROR_MSSG_7    "Observation number is out of range."
#define ERROR_MSSG_8    "No state with name @ found."
#define ERROR_MSSG_9    "No action with name @ found."
#define ERROR_MSSG_10   "No observation with name @ found."
#define ERROR_MSSG_11   "Bad probability. Must be between 0 and 1."
#define ERROR_MSSG_12   "Not enough matrix entries."
#define ERROR_MSSG_13   "Unreckognized matrix context."
#define ERROR_MSSG_14   "Duplicate name '@' in list."
#define ERROR_MSSG_15   "Too many matrix entries."
#define ERROR_MSSG_16   "Missing 'discount:' specification."
#define ERROR_MSSG_17   "Missing 'values:' specification"
#define ERROR_MSSG_18   "Missing 'states:' specification"
#define ERROR_MSSG_19   "Missing 'actions:' specification"
#define ERROR_MSSG_20   "Missing 'observations:' specification"
#define ERROR_MSSG_21   "Bad probability sum for T: @."
#define ERROR_MSSG_22   "Bad probability sum for O: @"
#define ERROR_MSSG_23   "Syntax error."
#define ERROR_MSSG_24   "Bad probability sum for start belief state"
#define ERROR_MSSG_25   "The keyword 'reset' is not allowed in this context"
#define ERROR_MSSG_26   "Observations not valid in MDP, or missing observations in POMDP"
#define ERROR_MSSG_27   "This type of starting state specification is not valid for an MDP."
#define ERROR_MSSG_28   "Not a valid reward syntax for this problem type."

/* Corresponding values for each error type */
#define ILL_CHAR_ERR                 0
#define BAD_DISCOUNT_VAL             1
#define BAD_NUM_STATES               2
#define BAD_NUM_ACTIONS              3
#define BAD_NUM_OBS                  4
#define BAD_STATE_VAL                5
#define BAD_ACTION_VAL               6
#define BAD_OBS_VAL                  7
#define BAD_STATE_STR                8
#define BAD_ACTION_STR               9
#define BAD_OBS_STR                 10
#define BAD_PROB_VAL                11
#define TOO_FEW_ENTRIES             12
#define BAD_MATRIX_CONTEXT          13
#define DUPLICATE_STRING            14
#define TOO_MANY_ENTRIES            15
#define MISSING_DISCOUNT            16
#define MISSING_VALUES              17
#define MISSING_STATES              18
#define MISSING_ACTIONS             19
#define MISSING_OBS                 20
#define BAD_TRANS_PROB_SUM          21
#define BAD_OBS_PROB_SUM            22
#define PARSE_ERR                   23
#define BAD_START_PROB_SUM          24
#define BAD_RESET_USAGE             25
#define OBS_IN_MDP_PROBLEM          26
#define BAD_START_STATE_TYPE        27
#define BAD_REWARD_SYNTAX           28

/* Miscellaneous messages */
#define OUT_OF_RANGE_MSSG "** ERROR ** ERR_enter: errorid of %d out of range\n"
#define NO_ERRORS	"No Errors found.\n"
#define NO_LINE_ERROR	"ERROR: "
#define LINE_ERROR	"ERROR:   line %d: "
#define NO_LINE_WARNING	"WARNING:"
#define LINE_WARNING	"WARNING: line %d: "

/*************************************************************************/
/******************************* TYPEDEFS  *******************************/
/*************************************************************************/

/*	The following structure definition is for the data structure that will
hold the errors generated by the compiler.  This structure is needed so that
a list of errors can be reported to the user while allowing as many errors
as possible to be detected by the compiler before notifying user.
	The data structure will be a linked list of the nodes described
below.
*/

typedef struct enode
{
	char *source;	/* what module and routine generated the error */

	int lineNumber;	/* line number in source code of error */

	int errorNumber;	/* which error was it */

	char *modString;	/* additional information for output, to
				be inserted into error message where the
				ERR_META character is found  */

	struct enode *nextError;	/* pointer to next node */
} Err_node;

extern int ERR_dump();
extern void ERR_enter();
extern void ERR_inititalize();

#endif
/*************************************************************************/
