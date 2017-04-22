CXX=g++
NVCC=nvcc
CXXFLAGS=-g -std=c++11 -O0 -flto -Wall -L/opt/cuda-8.0/lib64/ -lcudart $(shell pkg-config --cflags --libs libglfw)  #-stdlib=libstdc++
BIN=gravitysim

SRC=$(wildcard *.cpp)
# OBJ=$(SRC:%.cpp=%.o)
CU_SRC=cudaBHSpaceModel.o

mac: 
	$(CXX) $(CXXFLAGS) -Wl,-framework,OpenGL $(SRC) -o $(BIN)

linux: cudaBHSpaceModel.o
	$(CXX) $(CXXFLAGS) -lm -lGL $(SRC) $(CU_SRC) -o $(BIN)

cudaBHSpaceModel.o:
	$(NVCC) -std=c++11 -c cudaBHSpaceModel.cu -o $@

clean:
	rm -f *.o
	rm -f $(BIN)
