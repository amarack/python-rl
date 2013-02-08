# Makefile

PYTHON_INC_DIR = $(shell python-config --includes)
PYTHON_BUILD_FLAGS = $(shell python-config --ldflags)
NUMPY_INC = $(subst include,lib,$(PYTHON_INC_DIR))/site-packages/numpy/core/include/numpy/
CC	= gcc
CFLAGS	= -g -Wall -ansi #-pedantic
LDFLAGS = -bundle -flat_namespace -undefined suppress #-shared
TARGET  = libpomdp.so
SOURCES = $(shell echo *.c)
HEADERS = $(shell echo *.h)
OBJECTS = $(SOURCES:.c=.o)

PREFIX = $(DESTDIR)/usr/local
BINDIR = $(PREFIX)/bin

all: $(TARGET)

lex.o:		lex.c
		$(CC) $(CFLAGS) -c lex.c -o lex.o

lex.c:		scanner.l
		flex scanner.l
		cp lex.yy.c lex.c

bison.o:	bison.c
		$(CC) $(CFLAGS) -c bison.c -o bison.o

bison.c:	parser.y
		bison -d -v parser.y
		cp parser.tab.c bison.c
		cmp -s parser.tab.h tok.h || cp parser.tab.h tok.h

$(OBJECTS): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(PYTHON_INC_DIR) $(NUMPY_INC) -c $(SOURCES)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $(PYTHON_BUILD_FLAGS) -o $(TARGET) $(OBJECTS)

clean:
	rm -f *.o *~ lex.c lex.yy.c bison.c tok.h parser.tab.c parser.tab.h parser.output $(TARGET)

