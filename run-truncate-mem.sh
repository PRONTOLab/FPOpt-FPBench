#!/usr/bin/env sh
set -e

export CXXFLAGS='-fno-exceptions -fpass-plugin=/scr/ivan/src/Enzyme/enzyme/build-16.0.6.release/Enzyme/ClangEnzyme-16.so -Xclang -load -Xclang /scr/ivan/src/Enzyme/enzyme/build-16.0.6.release/Enzyme/ClangEnzyme-16.so -ffast-math -fuse-ld=lld -Rpass=enzyme'

#export CXXFLAGS="$CXXFLAGS -O0"
export CXXFLAGS="$CXXFLAGS -O2"

export CXXFLAGS="$CXXFLAGS -g"


export CXXFLAGS="$CXXFLAGS -include /scr/ivan/src/Enzyme/enzyme/Enzyme/Runtimes/FPRT/Trace.cpp"

#clang++ $CXXFLAGS /scr/ivan/src/Enzyme/enzyme/Enzyme/Runtimes/FPRT/Trace.cpp -O0 -c -o ./Trace.o
#export CXXFLAGS="$CXXFLAGS ./Trace.o"


#bash -x ./run-one.sh ./benchmarks/rosa.fpcore.c ./driver-enzyme.cpp
#bash -x ./run-one.sh ./benchmarks/fptaylor-tests.fpcore.c ./driver-enzyme.cpp

./run-all.sh ./driver-enzyme.cpp
