CXX=g++
CXXFLAGS=-g -std=c++11 -O0 -flto -Wall $(shell pkg-config --cflags --libs libglfw)  #-stdlib=libstdc++
BIN=gravitysim

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

mac: 
	$(CXX) $(CXXFLAGS) -Wl,-framework,OpenGL $(SRC) -o $(BIN)

linux:
	$(CXX) $(CXXFLAGS) -lm -lGL $(SRC) -o $(BIN)

clean:
	rm -f *.o
	rm -f $(BIN)
