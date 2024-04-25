#!/usr/bin/env sh
set -e

export CXXFLAGS='-fno-exceptions -fpass-plugin=/scr/ivan/src/Enzyme/enzyme/build-16.0.6.release/Enzyme/ClangEnzyme-16.so -Xclang -load -Xclang /scr/ivan/src/Enzyme/enzyme/build-16.0.6.release/Enzyme/ClangEnzyme-16.so -O3 -ffast-math -fuse-ld=lld'

export CXXFLAGS="$CXXFLAGS -g"

clang++ $CXXFLAGS /scr/ivan/src/Enzyme/enzyme/Enzyme/Runtimes/FPRT/Trace.cpp -O2 -c -o ./Trace.o

export CXXFLAGS="$CXXFLAGS ./Trace.o"

bash -x ./run-one.sh ./benchmarks/rosa.fpcore.c ./driver-enzyme.cpp
#./run-all.sh ./driver-enzyme.cpp
