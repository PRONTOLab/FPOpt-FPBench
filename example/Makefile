ENZYME_PATH ?= /home/brant/Enzyme/build/Enzyme/ClangEnzyme-15.so
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
		   -mllvm --fpopt-log-path=enzyme.txt \
		   -mllvm --fpopt-target-func-regex=Pendulum

SRC ?= example.c
LOGGER ?= fp-logger.cpp
EXE ?= example-logged.exe

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

example-fpopt.exe: example.cpp
	$(CXX) example.cpp $(CXXFLAGS) $(FPOPTFLAGS) -o $@

clean:
	rm -f $(EXE) example-logged.cpp example.cpp

