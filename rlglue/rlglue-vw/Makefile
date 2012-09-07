 LDFLAGS := $(LDFLAGS) -L/usr/local/lib -L../vowpalwabbit
 CFLAGS := $(CFLAGS) -I/usr/local/include/rlglue

LDFLAGS := $(LDFLAGS) -lm -lrlutils -lrlagent -lrlgluenetdev -l vw -l allreduce -l boost_program_options -l z -l pthread
CFLAGS := $(CFLAGS) -Wall -pedantic

BINARIES := SampleSarsaAgent

all: SampleSarsaAgent

SampleSarsaAgent: SampleSarsaAgent.o
	$(CXX)   SampleSarsaAgent.o $(LDFLAGS) -o SampleSarsaAgent

SampleSarsaAgent.o: SampleSarsaAgent.cpp
	$(CXX)  $(CFLAGS) -c SampleSarsaAgent.cpp -o SampleSarsaAgent.o

clean:
	rm -f  *.o $(BINARIES) *~