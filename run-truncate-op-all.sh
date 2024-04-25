#!/usr/bin/env sh
set -e

export CXXFLAGS='-include enzyme/fprt/mpfr.h -fno-exceptions -fpass-plugin=/scr/ivan/src/Enzyme/enzyme/build-16.0.6.release/Enzyme/ClangEnzyme-16.so -Xclang -load -Xclang /scr/ivan/src/Enzyme/enzyme/build-16.0.6.release/Enzyme/ClangEnzyme-16.so -mllvm --enzyme-truncate-all=11-52to5-18 -lmpfr -O3 -ffast-math -fuse-ld=lld'

./run-one.sh ./benchmarks/rosa.fpcore.c ./driver.cpp
 #./run-all.sh ./driver.cpp
