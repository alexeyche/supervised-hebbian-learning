all: shl

shl: shl.o
	clang++ -Wall shl.o -o libshl.so -shared 

shl.o: shl.cpp
	clang++ -Ieigen -std=c++14 -o0 -g -c shl.cpp -o shl.o -fPIC 

clean:
	rm -rf *.o libshl.so