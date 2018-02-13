all: shl

shl: shl.o
	g++ -Wall shl.o -o libshl.so -shared 

shl.o: shl.cpp
	g++ -Ieigen -c shl.cpp -fPIC

clean:
	rm -rf *.o libshl.so