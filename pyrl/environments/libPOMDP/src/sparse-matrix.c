/* sparse-matrix.c

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

   This module will implementall the routines that pertain to creating and
   manipulating a sparse representation of a matrix.

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mdp-common.h"
#include "sparse-matrix.h"

/**********************************************************************/
/********************  Routines for row linked lists  *****************/
/**********************************************************************/
I_Matrix_Row_Node 
newRowNode( int col, double value ) {
/*
   Just allocates memory and sets the fields.
*/
  I_Matrix_Row_Node new_node;

  new_node = (I_Matrix_Row_Node ) XMALLOC( sizeof( *new_node ));
  new_node->column = col;
  new_node->value = value;
  new_node->next = NULL;

  return( new_node );
  
}  /* newRowNode */
/**********************************************************************/
void 
destroyRow( I_Matrix_Row_Node row ) {
  I_Matrix_Row_Node temp;

  while( row != NULL ) {
    temp = row;
    row = row->next;
    XFREE( temp );
  } /* while */
}  /* destroyRow */
/**********************************************************************/
I_Matrix_Row_Node 
removeRowNode( I_Matrix_Row_Node row, 
				int col, int *count ) {
/*
   Remove the node with column == col, if it exists.  It decrements
   the count if it is removed.  If there is no entry for this in the
   list then we simply igore it and return the original list.
*/
  I_Matrix_Row_Node temp_node, cur_node, trail_node;

  /* Case if list is empty */
  if( row == NULL )
    return( NULL );

  /* Case of removing first item in the list */
  if( row->column == col ) {
    temp_node = row;
    row = row->next;
    XFREE( temp_node );
    (*count)--;
    return( row );
  }  /* if we should remove first node */

  trail_node = row;
  cur_node = row->next;
  while( cur_node != NULL ) {

    /* Then bingo!  We found the one to remove */
    if( cur_node->column == col ) {
      trail_node->next = cur_node->next;
      XFREE( cur_node );
      (*count)--;
      return( row );
    }  /* if we found the one to remove */

    trail_node = cur_node;
    cur_node = cur_node->next;
  }  /* while */

  /* Otherwise we did not find an entry with this column, so just */
  /* return the list and do nothing. */
  return( row );

}  /* removeRowNode */
/**********************************************************************/
I_Matrix_Row_Node 
addEntryToRow( I_Matrix_Row_Node row, 
	       int col, double value,
	       int *count, int accumulate ) {
  /*
   Will step through the linked list until either the column is found
   or until we see a strictly smaller column number.  It will either
   replace the current value (if the column already exists) or insert a
   new node into the list just before the first column number that is
   highest.  This way, the list always remains sorted by column
   number.  The pointer to the int 'count' will be incremented only if
   we add it to the list, not when we simply replace a value.  *count 
   should be the length of the list sent in, but this routine does not
   check to make sure it is.

   The accumulate flag will specify whether we should replace the 
   current value (if any) or add the value to the existing value.
   This can be used to build up probabilities in a matrix piecemeal.
   */
  I_Matrix_Row_Node new_node, cur_node, trail_node;
  
  /* Case if we attempt to add a zero entry.  We need to see if there */
  /* is already a non-zero entry for it in the list, and if so remove */
  /* it.  Otherwise we simply ignore this entry request, since we are */
  /* using a sparse representation. */
  if( IS_ZERO( value ) ) {
    if ( accumulate )
      return ( row );
    else
      return( removeRowNode( row, col, count ));
  }

  /* Case if the list is empty */
  if( row == NULL ) {
    new_node = newRowNode( col, value );
    (*count)++;
    return( new_node );
  }  /* if row empty */

  /* Case if it should be inserted into front of list */
  if( row->column > col ) {
    new_node = newRowNode( col, value );
    new_node->next = row;
    (*count)++;
    return( new_node );
  }  /* if should be inserted in front of list */

  cur_node = row;
  while( cur_node != NULL ) {
    
    /* Case if we should simply replace (or accumulate) 
       the current value */
    if( cur_node->column == col ) {
      
      if( accumulate )
	cur_node->value += value;
      else
	cur_node->value = value;

      return( row );
    } /* if we should just replace value */

    /* Case if this is the point we should insert it (i.e. between */
    /* trail_node and cur_node */
    if( cur_node->column > col ) {
      new_node = newRowNode( col, value );
      trail_node->next = new_node;
      new_node->next = cur_node;
      (*count)++;
      return( row );
    }  /* if we have found place to insert it */

    trail_node = cur_node;
    cur_node = cur_node->next;
  }  /* while */

  /* Case if it should be inserted at the end of the list.  This */
  /* should be the only case where we exit the bottom of the loop, all */
  /* other cases should exit the routine from within the loop */

  new_node = newRowNode( col, value );
  trail_node->next = new_node;
  (*count)++;
  return( row );

}  /* addEntryToRow */
/**********************************************************************/
void 
displayRow( I_Matrix_Row_Node row ) {

  if( row == NULL )
    printf( "<empty>");

  while( row != NULL ) {
    printf("[%d] %.3lf ", row->column, row->value );
    row = row->next;
  }  /* while */

  printf( "\n");
}  /* displayRow */

/**********************************************************************/
/********************  Routines for intermediate matrix    ************/
/**********************************************************************/
I_Matrix 
newIMatrix( int num_rows ) {
  I_Matrix i_matrix;

  i_matrix = (I_Matrix) XMALLOC( sizeof( *i_matrix));
  
  i_matrix->num_rows = num_rows;

  i_matrix->row = (I_Matrix_Row_Node *)
    XCALLOC( num_rows, sizeof( *(i_matrix->row) ));

  i_matrix->row_length = (int *) XCALLOC( num_rows, sizeof( int ));

  return( i_matrix );
}  /* newIMatrix */
/**********************************************************************/
void 
destroyIMatrix( I_Matrix i_matrix ) {
  int i;

  XFREE( i_matrix->row_length );

  for( i = 0; i < i_matrix->num_rows; i++ )
    destroyRow( i_matrix->row[i] );
  XFREE( i_matrix->row );

  XFREE( i_matrix );

}  /* destroyIMatrix */
/**********************************************************************/
int 
addEntryToIMatrix( I_Matrix i_matrix, int row, 
		   int col, double value ) {

  assert(( i_matrix != NULL) 
	 && (row >=0) && ( row < i_matrix->num_rows ));

  i_matrix->row[row] = addEntryToRow( i_matrix->row[row], col, value, 
				     &(i_matrix->row_length[row]), 0 );

  return( 1 );
}  /* addEntryToIMatrix */
/**********************************************************************/
int 
accumulateEntryInIMatrix( I_Matrix i_matrix, int row, 
			  int col, double value ) {
/*
   This routine is the same as addEntryToIMatrix() except it will call 
   the addEntryToRow() routine with the accumulate flag set.  Thus, if
   there is a value already at this row and column, the new value will 
   be the sum of the old and the new value.
*/
  assert(( i_matrix != NULL) 
	 && (row >=0) && ( row < i_matrix->num_rows ));

  i_matrix->row[row] = addEntryToRow( i_matrix->row[row], col, value, 
				     &(i_matrix->row_length[row]), 1 );

  return( 1 );
}  /* addEntryToIMatrix */
/**********************************************************************/
int 
countEntriesInIMatrix( I_Matrix i_matrix ) {
  int i;
  int total = 0;

  for( i = 0; i < i_matrix->num_rows; i++ )
    total += i_matrix->row_length[i];

  return( total );
}  /* countEntriesInIMatrix */
/**********************************************************************/
double 
sumIMatrixRowValues( I_Matrix i_matrix, int row ) {
  double sum = 0.0;
  I_Matrix_Row_Node list;

  list = i_matrix->row[row];

  while( list != NULL ) {
    sum += list->value;
    list = list->next;
  }  /* while */
  return( sum );
}  /* sumIMatrixRowValues */
/**********************************************************************/
void 
displayIMatrix( I_Matrix i_matrix ) {
  int i;
  
  for( i = 0; i < i_matrix->num_rows; i++ ) {
    printf( "(len=%d, sum =%.1lf)Row=%d: ", i_matrix->row_length[i], 
	   sumIMatrixRowValues( i_matrix, i ), i );
    displayRow( i_matrix->row[i] );

  }  /* for i */

} /* displayIMatrix */

/**********************************************************************/
/********************  Routines for sparse matrix    ******************/
/**********************************************************************/
Matrix 
newMatrix( int num_rows, int num_non_zero ) {
  Matrix matrix;

  matrix = (Matrix) XMALLOC( sizeof( *matrix ));

  matrix->num_rows = num_rows;
  matrix->num_non_zero = num_non_zero;

  matrix->mat_val = (double *) 
    XCALLOC( num_non_zero, sizeof( double ));
  matrix->col = (int *) 
    XCALLOC( num_non_zero, sizeof( int ));
  matrix->row_start = (int *) 
    XCALLOC( num_rows, sizeof( int ));
  matrix->row_length = (int *) 
    XCALLOC( num_rows, sizeof( int ));

  return( matrix );
}  /* newMatrix */
/**********************************************************************/
void 
destroyMatrix( Matrix matrix ) {

  XFREE( matrix->row_length );
  XFREE( matrix->row_start );
  XFREE( matrix->col );
  XFREE( matrix->mat_val );

  XFREE( matrix );
}  /* destroyMatrix */
/**********************************************************************/
Matrix 
transformIMatrix( I_Matrix i_matrix ) {
/*
   Will convert a matrix object in intermediate form into a sparse
   representation.  It does not free the memory of the intermediate
   representation.
   */
  Matrix matrix;
  int row;
  I_Matrix_Row_Node cur_node;
  int index = 0;  /* holds the position in the mat_val and col arrays */

  /* Allocate the appropriate amount of memory */
  matrix = newMatrix( i_matrix->num_rows, 
		     countEntriesInIMatrix( i_matrix ));

  /* Now go through and set the values */
  for( row = 0; row < i_matrix->num_rows; row++ ) {

    matrix->row_start[row] = index;
    matrix->row_length[row] = i_matrix->row_length[row];
    
    cur_node = i_matrix->row[row];
    while( cur_node != NULL ) {

      matrix->col[index] = cur_node->column;
      matrix->mat_val[index] = cur_node->value;
      index++;

      cur_node = cur_node->next;
    }  /* while cur_node */

  }  /* for row */
  
  assert( index == matrix->num_non_zero );

  return( matrix );
}  /* transformIMatrix */
/**********************************************************************/
double 
sumRowValues( Matrix matrix, int row ) {
  double sum = 0.0;
  int col;

  for( col = matrix->row_start[row];
      col < matrix->row_start[row] + matrix->row_length[row];
      col++ )
    sum += matrix->mat_val[col];

  return( sum );
}  /* sumRowValues */
/**********************************************************************/
void 
displayMatrix( Matrix matrix ) {
  int i, j;
  
  for( i = 0; i < matrix->num_rows; i++ ) {
    printf( "(len=%d, sum=%.1lf)Row=%d: ", matrix->row_length[i], 
	   sumRowValues( matrix, i ), i );

    
    if( matrix->row_length[i] == 0 )
      printf( "<empty>");
    
    for( j = matrix->row_start[i];
	j < matrix->row_start[i] + matrix->row_length[i];
	j++ )
      printf("[%d] %.3lf ", matrix->col[j], matrix->mat_val[j] );
    
    printf( "\n");
    
  }  /* for i */

} /* displayMatrix */
/**********************************************************************/
double 
getEntryMatrix( Matrix matrix, int row, int col ) {
/*
   Returns the value for a particular entry in the matrix.  It looks in
   the row for the specified column, and if it is not found it will
   return zero, since after all this is a sparse representation.
*/
  int j;

  for( j = matrix->row_start[row];
      j < matrix->row_start[row] + matrix->row_length[row];
      j++ )

    if( matrix->col[j] == col )
      return( matrix->mat_val[j] );
 
  return( 0.0 );
}  /* getEntryMatrix */
/**********************************************************************/



