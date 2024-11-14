HOME ?= /home/vimarsh6739/
ENZYME_PATH ?= $(HOME)/sync/Enzyme/build/Enzyme/ClangEnzyme-15.so
LLVM_PATH ?= $(HOME)/llvms/llvm15/build/bin
CXX = $(LLVM_PATH)/clang++

CXXFLAGS = -O3 -Wall \
		   -I$(HOME)/tools/include \
           -L$(HOME)/tools/lib \
           -I /usr/include/c++/11 \
           -I /usr/include/x86_64-linux-gnu/c++/11 \
           -L /usr/lib/gcc/x86_64-linux-gnu/11 \
           -fno-exceptions \
           -fpass-plugin=$(ENZYME_PATH) \
           -Xclang -load -Xclang $(ENZYME_PATH) \
           -lmpfr -ffast-math -fno-finite-math-only -fuse-ld=lld

FPOPTFLAGS += -mllvm --enzyme-enable-fpopt \
		   -mllvm --enzyme-print-herbie \
		   -mllvm --enzyme-print-fpopt \
		   -mllvm --fpopt-log-path=example.txt \
		   -mllvm --fpopt-target-func-regex=example \
		   -mllvm --fpopt-enable-solver \
		   -mllvm --fpopt-enable-pt \
		   -mllvm --fpopt-comp-cost-budget=0 \
		   -mllvm --fpopt-num-samples=1000 \
		   -mllvm --fpopt-show-table \
		   -mllvm --fpopt-cost-model-path=../microbm/cm.csv

ABLATIONFLAGS += -mllvm --fpopt-extra-ifconv

SRC ?= example.c
LOGGER ?= fp-logger.cpp
EXE ?= example.exe example-logged.exe example-golden.exe example-fpopt.exe

.PHONY: all clean

all: $(EXE)

example.cpp: $(SRC)
	python3 fpopt-original-driver-generator.py $(SRC) example

example-golden.cpp: $(SRC)
	python3 fpopt-golden-driver-generator.py $(SRC) 128 example

example-logged.cpp: $(SRC)
	python3 fpopt-logged-driver-generator.py $(SRC) example

example.exe: example.cpp
	$(CXX) example.cpp $(CXXFLAGS) -o $@

example-golden.exe: example-golden.cpp
	$(CXX) example-golden.cpp $(CXXFLAGS) -lmpfr -o $@

example-logged.exe: example-logged.cpp $(LOGGER)
	$(CXX) $(LOGGER) example-logged.cpp $(CXXFLAGS) -mllvm --enzyme-inline=0 -o $@

example.txt: example-logged.exe
	./example-logged.exe > $@

example-fpopt.exe: example.cpp example.txt
	$(CXX) example.cpp $(CXXFLAGS) $(ABLATIONFLAGS) $(FPOPTFLAGS) -o $@

clean:
	rm -f $(EXE) example-logged.cpp example.cpp example-golden.cpp example.txt

