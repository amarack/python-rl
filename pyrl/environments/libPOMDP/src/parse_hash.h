/* parse_hash.h 

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

#ifndef MDP_PARSE_HASH_H
#define MDP_PARSE_HASH_H 1

#define HASH_TABLE_SIZE      255

typedef enum { nt_state, nt_action, 
               nt_observation, nt_unknown } Mnemonic_Type;

typedef struct Node_Struct *Node;
struct Node_Struct {
   Mnemonic_Type type;
   int number;
   char *str;
   Node next;
};

extern void H_create();
extern void H_destroy();
extern int H_enter( char *str, Mnemonic_Type type );
extern int H_lookup( char *str, Mnemonic_Type type );

#endif
