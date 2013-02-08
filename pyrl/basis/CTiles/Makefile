
# Makefile for compiling CTiles which should work on Mac or Linux
# Additional changes were made from Rich Sutton's original to allow it to work
# with any recent version of python.

PYTHON_INC_DIR = $(shell python-config --includes)
PYTHON_BUILD_FLAGS = $(shell python-config --ldflags)

.PHONY : tiles

tiles: tiles.so

tilesInt.o: tilesInt.C
	c++ -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -mno-fused-madd -fno-common -dynamic -DNDEBUG -g -O3 -Wall -Wstrict-prototypes $(PYTHON_INC_DIR) -c tilesInt.C -o tilesInt.o

tiles.o: tiles.cpp
	c++ -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -mno-fused-madd -fno-common -dynamic -DNDEBUG -g -O3 -Wall -Wstrict-prototypes $(PYTHON_INC_DIR) -c tiles.cpp -o tiles.o

tiles.so: tiles.o tilesInt.o
	c++ -Wl,-F. -Wl,-F. -bundle $(PYTHON_BUILD_FLAGS) tiles.o tilesInt.o -o tiles.so

clean:
	rm *.so *.o