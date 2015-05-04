CC = g++
CFLAGS= -g -Wall -std=c++0x

default: compile

compile: Main.o Net.o Node.o Ants.o readC4.o
	$(CC) $? -o nn

Main.o: Main.cpp
	$(CC) $(CFLAGS) -c $< -o $@

Net.o: Net.cpp Net.h
	$(CC) $(CFLAGS) -c $< -o $@

Node.o: Node.cpp Node.h
	$(CC) $(CFLAGS) -c $< -o $@

Ants.o: Ants.cpp Ants.h
	$(CC) $(CFLAGS) -c $< -o $@

readC4.o: readC4.cpp readC4.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o *~ nn
