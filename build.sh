#!/bin/bash
g++ -g -O0 -march=native -std=c++14 -ffast-math -Wall -Wfatal-errors $(pkg-config --libs opencv) -o test test.cpp
