#SHELL=/bin/csh

CPP = mpicxx
CPPFLAGS =-c -O3 -g -DFFTW -I/usr/local/include -I/home/mirco/Scrivania/p3dfft.3-master/
CC = mpicxx
CFLAGS =

# ----------------

p3dfft_test: p3dfft_test.o
	$(CPP) -o p3dfft_test p3dfft_test.o -L /home/mirco/Scrivania/p3dfft.3-master/ -lp3dfft.3 -g -L/usr/local/lib/ -lfftw3 -lfftw3f -lm

clean:
	/bin/rm *.o p3dfft_test
