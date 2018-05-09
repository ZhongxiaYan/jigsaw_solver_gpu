#!/bin/bash
CFLAGS="-g -O3 -march=native -std=c++14 -ffast-math -Wall -Wfatal-errors -Wno-unknown-pragmas -lboost_program_options $(pkg-config --cflags --libs opencv) -lstdc++"
mpic++ -o mpi mpi.cpp -fopenmp $CFLAGS -DNDEBUG