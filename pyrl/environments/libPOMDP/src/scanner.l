%{
/*  scanner.l

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

  This module contains the patterns and actions needed by the scanner
  generator "LEX".  
*/

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "mdp-common.h"
#include "parse_err.h"     /* Routines to keep track of parsing errors */
#include "parse_hash.h"    /* Routines to hold temporary symbols and strings */
#include "parse_constant.h"
#include "parser.h"         /* token values */

#define NUM_LETTERS		26
#define MAX_IDENT_LENGTH        32

#define TO_UPPER_CASE            1
#define NORMAL                   0

extern long currentLineNumber;
extern void ERR_enter();   /* assumes calling program
                              will initialize error list */

/* These must appear in consecutive order just like they are defined in 
   the reserved_token array below.
*/
#define NUM_RESERVED_WORDS              16
static char *reserved_str[] = {
   "discount",
   "values",
   "states",
   "actions",
   "observations",
   "T",
   "O",
   "R",
   "uniform",
   "identity",
   "reward",
   "cost",
   "start",
   "include",
   "exclude",
   "reset"
};
static int reserved_token[] = {
   DISCOUNTTOK,
   VALUESTOK,
   STATETOK,
   ACTIONTOK,
   OBSTOK,
   TTOK,
   OTOK,
   RTOK,
   UNIFORMTOK,
   IDENTITYTOK,
   REWARDTOK,
   COSTTOK,
   STARTTOK,
   INCLUDETOK,
   EXCLUDETOK,
   RESETTOK
};


/**********************************************************************/
void 
IntToYylval()
{
Constant_Block *aConst;

   aConst = (Constant_Block *) XMALLOC(sizeof(Constant_Block));
   aConst->theTag = CONST_INT;
   aConst->theValue.theInt = atoi(yytext);

   yylval.constBlk = aConst;
}
/**********************************************************************/
void 
FloatToYylval()
{
Constant_Block *aConst;

   aConst = (Constant_Block *) XMALLOC(sizeof(Constant_Block));
   aConst->theTag = CONST_FLOAT;
   aConst->theValue.theFloat = atof(yytext);

   yylval.constBlk = aConst;
}
/**********************************************************************/
void 
StringToYylval()
/*  This routine will copy a string constant from yytext to the to the 
yylval.  
*/
{
   Constant_Block *aConst;
   char *tempString;
   int i;

   tempString = (char *) XCALLOC(strlen(yytext)+1, sizeof(char));

   for (i = 0; i < yyleng; i++)
	    tempString[i] = yytext[i];

   tempString[i] = '\0';   /* add null terminator */

   aConst = (Constant_Block *) XMALLOC(sizeof(Constant_Block));
   aConst->theTag = CONST_STRING;
   aConst->theValue.theString = tempString;

   yylval.constBlk = aConst;

}  /*  */
/**********************************************************************/
int 
CheckReserved() {
  int i;

  for( i = 0; i < NUM_RESERVED_WORDS; i++ ) 
     if( strcmp( reserved_str[i], yytext ) == 0 )
        return( reserved_token[ i ] );
  
  return( -1 );

}  /* CheckReserved */
/**********************************************************************/
%}
/************************* Character Classes  *******************************/

Digit [0-9]
Letter [A-Za-z]
Other [\-\_]
IntLiteral {Digit}+

%%
{IntLiteral} 			{
                                          IntToYylval();
                                          return (INTTOK);  /* Integer Literal */
					}   

{IntLiteral}"."{IntLiteral}	{
                                          FloatToYylval();
                                          return (FLOATTOK);  /* Floating Point Literal */
					}  

{Letter}({Digit}|{Letter}|{Other})*	{
                                  int tok_val;
                                  tok_val = CheckReserved();
                                  if( tok_val < 0 ) {
                                     StringToYylval();
                                     return (STRINGTOK);
                                  }
                                  else
                                     return( tok_val );
 				}

\#[^\n]*			;   /* Comment: No action */


"*"				{         
                                   return (ASTERICKTOK);
                                }
"-"				{         
                                   return (MINUSTOK);
                                }
"+"				{         
                                    return (PLUSTOK);
                                }
":"				{         
                                    return (COLONTOK);
                                }

" "				; /* Spacedelimiter: No action */ 

\t				; /* Tab delimiter: No action */

\r				; /* Carriage return: No action */

\n				{  
                                    currentLineNumber++; /* Newline delimiter */
                                }


.				{ 
                                  
                                  ERR_enter("Scanner<yylex>",currentLineNumber,
                                            ILL_CHAR_ERR, yytext);
                                }








