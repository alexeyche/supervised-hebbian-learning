all: shl

shl: shl.o
	clang++ -Wall shl.o -o libshl.so -shared 

shl.o: shl.cpp
	clang++ -DEIGEN_STACK_ALLOCATION_LIMIT=1000000 -Ieigen -std=c++14 -O3 -g -c shl.cpp -o shl.o -fPIC -ffast-math -mavx2 -msse2

clean:
	rm -rf *.o libshl.so