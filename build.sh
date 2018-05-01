#!/bin/bash
g++ -g -O1 -std=c++14 -Wall -Wfatal-errors $(pkg-config --libs opencv) -o test test.cpp
