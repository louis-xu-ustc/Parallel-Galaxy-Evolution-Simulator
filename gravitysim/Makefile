EXECUTABLE := gravitysim

CU_FILES   := cudaBHSpaceModel.cu

CU_DEPS    :=

CC_FILES   := $(wildcard *.cpp)

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-g -std=c++11 -O0 -flto -Wall $(shell pkg-config --cflags --libs libglfw)
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

# normal mode
NVCCFLAGS=-std=c++11 -O3 -m64 --gpu-architecture compute_35
# debug mode
#NVCCFLAGS=-O0 -g -G -m64 --gpu-architecture compute_35

LIBS += GL glut cudart

ifneq ($(wildcard /opt/cuda-8.0/.*),)
# Latedays
LDFLAGS=-L/opt/cuda-8.0/lib64/ -lcudart
else
# GHC
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/Object.o $(OBJDIR)/Perf.o $(OBJDIR)/QuadTree.o \
     $(OBJDIR)/Report.o $(OBJDIR)/Screen.o $(OBJDIR)/SpaceController.o $(OBJDIR)/SpaceModel.o \
     $(OBJDIR)/SpaceView.o $(OBJDIR)/FMMSpaceModel.o $(OBJDIR)/cudaBHSpaceModel.o \
     $(OBJDIR)/BHSpaceModel.o $(OBJDIR)/basic_types.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

check:	default
		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@


