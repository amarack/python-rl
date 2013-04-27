/* sparse-matrix.h 

   Header file for sparse-matrix.c.

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
#ifndef MDP_SPARSE_MATRIX_H
#define MDP_SPARSE_MATRIX_H 1

#define POS_ZERO_TOLERANCE        0.0000000001
#define NEG_ZERO_TOLERANCE        -0.0000000001

#define IS_ZERO(X)   ((X < POS_ZERO_TOLERANCE) && ( X > NEG_ZERO_TOLERANCE ))

/*  Each row of the intermediate form of the matrix will consist
    of a linked list.  This structure is for each node of those linked
    lists.  The pertinent information is the column number for the
    entry and the value.  Additionally, this linked list will be kept
    sorted by column so when adding an element we look for the column
    and insert it in place. It will be sorted from least to greatest.
    */ 
typedef struct I_Matrix_Row_Node_Struct *I_Matrix_Row_Node;
struct I_Matrix_Row_Node_Struct {
  int column;
  double value;
  I_Matrix_Row_Node next;
};

/*  A matrix in intermediate form will be a linked list for each row
    and a count of the number of non-zero entries for each row.
    */
struct I_Matrix_Struct {
  int num_rows;
  I_Matrix_Row_Node *row;   /* An array of pointers for the head of
			       each row's linked list */
  int *row_length;            /* An array for the current lengths of
			       each row. */
};
typedef struct I_Matrix_Struct *I_Matrix;

/* A matrix will be sparsely represented by a bunch of arrays
   */
struct Matrix_Struct {
  int num_rows;
  int num_non_zero;
  double *mat_val;  /* The actual non-zero entries stored row by row. */
  int *row_start;   /* the position for the start of each row in mat_val */
  int *row_length;  /* The length of each row in mat_val */
  int *col;         /* The column number for each entry in mat_val */
};
typedef struct Matrix_Struct *Matrix;

/**********************************************************************/
/******************************  External Routines  *******************/
/**********************************************************************/
extern void destroyRow( I_Matrix_Row_Node row );
extern I_Matrix_Row_Node addEntryToRow( I_Matrix_Row_Node row, 
				int col, double value,
				int *count, int accumulate );
extern void displayRow( I_Matrix_Row_Node row );

extern int addEntryToIMatrix( I_Matrix i_matrix, int row, 
			     int col, double value );
extern int accumulateEntryInIMatrix( I_Matrix i_matrix, int row, 
				    int col, double value );
extern void destroyIMatrix( I_Matrix i_matrix );
extern I_Matrix newIMatrix( int num_rows );
extern double sumIMatrixRowValues( I_Matrix i_matrix, int row );
extern Matrix newMatrix( int num_rows, int num_non_zero );
extern void destroyMatrix( Matrix matrix );
extern Matrix transformIMatrix( I_Matrix i_matrix );
extern void displayMatrix( Matrix matrix );
extern double sumRowValues( Matrix matrix, int row );
extern double getEntryMatrix( Matrix matrix, int row, int col );

#endif
