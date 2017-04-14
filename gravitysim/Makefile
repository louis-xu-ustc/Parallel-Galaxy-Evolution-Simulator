CXX=clang++
CXXFLAGS=-g -std=c++11 -O3 -flto -Wall $(shell pkg-config --cflags --libs libglfw)
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
