/* A Bison parser, made by GNU Bison 2.1.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     INTTOK = 1,
     FLOATTOK = 2,
     COLONTOK = 3,
     MINUSTOK = 4,
     PLUSTOK = 5,
     STRINGTOK = 6,
     ASTERICKTOK = 7,
     DISCOUNTTOK = 8,
     VALUESTOK = 9,
     STATETOK = 10,
     ACTIONTOK = 11,
     OBSTOK = 12,
     TTOK = 13,
     OTOK = 14,
     RTOK = 15,
     UNIFORMTOK = 16,
     IDENTITYTOK = 17,
     REWARDTOK = 18,
     COSTTOK = 19,
     RESETTOK = 20,
     STARTTOK = 21,
     INCLUDETOK = 22,
     EXCLUDETOK = 23,
     EOFTOK = 258
   };
#endif
/* Tokens.  */
#define INTTOK 1
#define FLOATTOK 2
#define COLONTOK 3
#define MINUSTOK 4
#define PLUSTOK 5
#define STRINGTOK 6
#define ASTERICKTOK 7
#define DISCOUNTTOK 8
#define VALUESTOK 9
#define STATETOK 10
#define ACTIONTOK 11
#define OBSTOK 12
#define TTOK 13
#define OTOK 14
#define RTOK 15
#define UNIFORMTOK 16
#define IDENTITYTOK 17
#define REWARDTOK 18
#define COSTTOK 19
#define RESETTOK 20
#define STARTTOK 21
#define INCLUDETOK 22
#define EXCLUDETOK 23
#define EOFTOK 258




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 131 "parser.y"
typedef union YYSTYPE {
  Constant_Block *constBlk;
  int i_num;
  double f_num;
} YYSTYPE;
/* Line 1447 of yacc.c.  */
#line 96 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;



