#!/bin/bash
CFLAGS="-g -O3 -march=native -std=c++14 -ffast-math -Wall -Wfatal-errors -Wno-unknown-pragmas $(pkg-config --cflags --libs opencv) -DNDEBUG"
g++ -o test test.cpp $CFLAGS
g++ -o test-omp test.cpp -fopenmp $CFLAGS
g++ -o test-demo test.cpp -fopenmp $CFLAGS -DDEMO
