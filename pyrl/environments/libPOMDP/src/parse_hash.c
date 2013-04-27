/* parse_hash.c 

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
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mdp-common.h"
#include "mdp.h"
#include "parse_hash.h"

Node *Hash_Table;

/**********************************************************************/
void 
H_create() {

   Hash_Table = (Node *) XCALLOC( HASH_TABLE_SIZE , sizeof( *Hash_Table ));

   /* Need these to start at zero! */
   gNumStates = gNumActions = gNumObservations = 0;
}  /* H_init */
/**********************************************************************/
void 
H_destroy() {
   Node temp;
   int i;

   for( i = 0; i < HASH_TABLE_SIZE; i++) 
      while( Hash_Table[i] != NULL ) {
         temp = Hash_Table[i];
         Hash_Table[i] = temp->next;
         XFREE( temp->str );
         XFREE( temp );
      }  /* while */
  
   XFREE( Hash_Table );
}  /* H_destroy */
/**********************************************************************/
int 
H_string( char *str ) {
   int max;

   if(( str == NULL) || (str[0] == '\0' )) {
      fprintf( stderr, "**ERR: Bad string in H_string().\n");
      exit( -1);
   }

   max = strlen( str ) - 1;

   switch( max ) {
   case 0:
      return( str[0] % HASH_TABLE_SIZE);
   case 1:
      return( ( str[0] * str[1] ) % HASH_TABLE_SIZE);
   case 2:
      return( ( str[0] * str[1] + str[2] ) % HASH_TABLE_SIZE);
   default:
      return( ( str[0] * str[1] * str[max-1] + str[max] ) % HASH_TABLE_SIZE);

   }  /* switch */

}  /* H_string */
/**********************************************************************/
int 
H_match( char *str, Mnemonic_Type type, Node node ) {

   if( node == NULL )  {
      fprintf( stderr, "**ERR: Null node in H_match().\n");
      exit(-1);
   }
      
   if( type != node->type )
      return 0;
   if( strcmp( str, node->str ) != 0 )
      return 0;
   
   return 1;
}  /* H_match */
/**********************************************************************/
int 
H_enter( char *str, Mnemonic_Type type ) {
   Node trail, temp;
   int hash;

   if(( str == NULL) || ( str[0] ==  '\0' )) {
      fprintf( stderr, "**ERR: Bad string in H_enter().\n");
      exit( -1);
   }

   hash = H_string( str );

   /* Find end of linked list */
   trail = temp = Hash_Table[hash];
   while( temp != NULL ) {
      trail = temp;
      /* if already in the list then there's a duplicate */
      if ( H_match( str, type, temp) == 1 ) 
         return 0;
      temp = temp->next;
   }  /* while */

   /* create node and set fields */
   temp = (Node) XMALLOC( sizeof(*temp));
   temp->next = NULL;
   temp->type = type;
   temp->str = (char *) XMALLOC( (strlen( str )+1) * sizeof (char ));
   strcpy( temp->str, str );

   /* Set number and increment appropriate value */
   switch( type ) {
   case nt_state:
      temp->number = gNumStates++;
      break;
   case nt_action:      
      temp->number = gNumActions++;
      break;
   case nt_observation:
      temp->number = gNumObservations++;
      break;
   default:
      fprintf( stderr, "**ERR: Bad type in H_enter()\n");
      exit( -1);
   }

   /* Add to hash table */
   if( trail == NULL ) 
      Hash_Table[hash] = temp;
   else
      trail->next = temp;

   return 1;
}  /* H_enterString */
/**********************************************************************/
int 
H_lookup( char *str, Mnemonic_Type type ) {
   int hash;
   Node temp;

   if(( str == NULL) || (str[0] ==  '\0' )) {
      fprintf( stderr, "**ERR: Bad string in H_getNum().\n");
      exit( -1);
   }

   hash = H_string( str );

   /* Find end of linked list */
   temp = Hash_Table[hash];
   while( temp != NULL ) {
      /* if already in the list then there's a duplicate */
      if ( H_match( str, type, temp) == 1 ) 
         return temp->number;
      temp = temp->next;
   }  /* while */

   return -1;

}  /* H_getNum */
/**********************************************************************/
