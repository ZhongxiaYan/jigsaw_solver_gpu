#!/bin/bash
CFLAGS="-g -O3 -march=native -std=c++14 -ffast-math -Wall -Wfatal-errors -Wno-unknown-pragmas $(pkg-config --libs opencv)"
g++ $CFLAGS -o test test.cpp
g++ $CFLAGS -fopenmp -o test-omp test.cpp
