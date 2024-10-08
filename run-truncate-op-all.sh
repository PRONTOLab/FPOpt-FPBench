#!/usr/bin/env sh
set -e

export CXXFLAGS='-I/home/brant/include -L/home/brant/lib -I /usr/include/c++/11 -I /usr/include/x86_64-linux-gnu/c++/11 -L /usr/lib/gcc/x86_64-linux-gnu/11 -include enzyme/fprt/mpfr.h -fno-exceptions -fpass-plugin=/home/brant/Enzyme/build/Enzyme/ClangEnzyme-14.so -Xclang -load -Xclang /home/brant/Enzyme/build/Enzyme/ClangEnzyme-14.so -mllvm --enzyme-truncate-all=11-52to5-18 -lmpfr -O3 -ffast-math -fuse-ld=lld'

./run-one.sh ./benchmarks/rosa.fpcore.c ./driver.cpp
 #./run-all.sh ./driver.cpp
