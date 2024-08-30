#!/usr/bin/env sh
set -e

export CXXFLAGS='-I/home/brant/include -L/home/brant/lib -I /usr/include/c++/11 -I /usr/include/x86_64-linux-gnu/c++/11 -L /usr/lib/gcc/x86_64-linux-gnu/11 -fno-exceptions -lmpfr -O3 -ffast-math -fuse-ld=lld'

./run-one-single-main.sh ./benchmarks/rosa.fpcore.c ./driver-fpopt-herbie.cpp 2>&1 | tee ./logs/rosa-original.txt
 #./run-all.sh ./driver.cpp
