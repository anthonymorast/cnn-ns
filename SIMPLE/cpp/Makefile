

OBJS=visual.o view.o drivencavity.o

LFLAGS=-g -Ofast
CFLAGS=-std=c++11 -g -Wall -Ofast

INCLUDES=-I.

all: simple

simple: $(OBJS)
	g++ -o simple $(OBJS) $(LFLAGS) 

.cc.o:
	g++ $(INCLUDES) $(CFLAGS) -c $<


clean:
	rm -f *.o lacart laforest


# make depend will create a file ".depend" with all the dependencies

depend: 
	rm -f .depend 
	g++ $(CFLAGS) $(DEFINES) $(INCLUDES) $(DEPENDFLAGS) $(OBJS:.o=.cc) > .depend



# if we have a .depend file, include it

ifeq (.depend,$(wildcard .depend))
include .depend
endif
