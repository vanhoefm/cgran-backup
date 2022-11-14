CC=gcc
CFLAGS=-g

ber: ber.o
	gcc $(LIBS) -ober ber.o

ber.o: ber.c
	gcc $(CFLAGS) -c ber.c

clean:
	rm -f ber *[.]o *~
