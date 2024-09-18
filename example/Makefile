ENZYME_PATH ?= /home/brant/sync/Enzyme/build/Enzyme/ClangEnzyme-15.so
LLVM_PATH ?= /home/brant/llvms/llvm15/build/bin
CXX = $(LLVM_PATH)/clang++

CXXFLAGS = -I/home/brant/include \
           -L/home/brant/lib \
           -I /usr/include/c++/11 \
           -I /usr/include/x86_64-linux-gnu/c++/11 \
           -L /usr/lib/gcc/x86_64-linux-gnu/11 \
           -fno-exceptions \
           -fpass-plugin=$(ENZYME_PATH) \
           -Xclang -load -Xclang $(ENZYME_PATH) \
           -lmpfr -O3 -ffast-math -fuse-ld=lld

FPOPTFLAGS += -mllvm --enzyme-enable-fpopt \
		   -mllvm --enzyme-print-herbie \
		   -mllvm --enzyme-print-fpopt \
		   -mllvm --fpopt-log-path=example.txt \
		   -mllvm --fpopt-target-func-regex=Pendulum \
		   -mllvm --fpopt-enable-solver \
		   -mllvm --fpopt-comp-cost-budget=1000

SRC ?= example.c
LOGGER ?= fp-logger.cpp
EXE ?= example.exe example-logged.exe example-fpopt.exe

.PHONY: all clean

all: $(EXE)

example.cpp: $(SRC)
	python3 fpopt-original-driver-generator.py $(SRC) "Pendulum"

example-logged.cpp: $(SRC)
	python3 fpopt-logged-driver-generator.py $(SRC) "Pendulum"

example.exe: example.cpp
	$(CXX) -Wall -O3 example.cpp $(CXXFLAGS) -o $@

example-logged.exe: example-logged.cpp $(LOGGER)
	$(CXX) -Wall -O3 $(LOGGER) example-logged.cpp $(CXXFLAGS) -o $@

example.txt: example-logged.exe
	./example-logged.exe > $@

example-fpopt.exe: example.cpp example.txt
	$(CXX) example.cpp $(CXXFLAGS) $(FPOPTFLAGS) -o $@

clean:
	rm -f $(EXE) example-logged.cpp example.cpp

