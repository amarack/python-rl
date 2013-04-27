/*
 *<SOURCE_HEADER>
 *
 *  <NAME>
 *    global.h
 *  </NAME>
 *  <AUTHOR>
 *    Anthony R. Cassandra
 *  </AUTHOR>
 *  <CREATE_DATE>
 *    July, 1998
 *  </CREATE_DATE>
 *
 *  <RCS_KEYWORD>
 *    $RCSfile: mdp-common.h,v $
 *    $Source: /u/cvs/proj/pomdp-solve/src/mdp/mdp-common.h,v $
 *    $Revision: 1.1 $
 *    $Date: 2004/10/10 03:44:59 $
 *  </RCS_KEYWORD>
 *
 *  <COPYRIGHT>
 *
 *    1994-1997, Brown University
 *    1998-2003, Anthony R. Cassandra
 *
 *    All Rights Reserved
 *                          
 *    Permission to use, copy, modify, and distribute this software and its
 *    documentation for any purpose other than its incorporation into a
 *    commercial product is hereby granted without fee, provided that the
 *    above copyright notice appear in all copies and that both that
 *    copyright notice and this permission notice appear in supporting
 *    documentation.
 * 
 *    ANTHONY CASSANDRA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 *    INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
 *    PARTICULAR PURPOSE.  IN NO EVENT SHALL ANTHONY CASSANDRA BE LIABLE FOR
 *    ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 *    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 *    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 *    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 *  </COPYRIGHT>
 *
 *</SOURCE_HEADER>
 */

/*
 *   Header file for all globally defined items in the mdp library.
 */

#ifndef MDP_COMMON_H
#define MDP_COMMON_H

#ifdef DMALLOC

#include "dmalloc.h"

#define XCALLOC(num, size) calloc( (num), (size) )
#define XMALLOC(size) malloc( size )
#define XREALLOC(p, size) realloc( (p), (size) )
#define XFREE(stale) free(stale)

#else

#define XCALLOC(num, size) calloc( (num), (size) )
#define XMALLOC(size) malloc( size )
#define XREALLOC(p, size) realloc( (p), (size) )
#define XFREE(stale) free(stale)

#endif

#endif

