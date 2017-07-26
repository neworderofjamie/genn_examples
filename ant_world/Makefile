
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_30 -std=c++11 -O3

all: ant_world

ant_world: ant_world.o
	$(CXX) -o ant_world ant_world.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lglfw -lGL -lGLU -lGLEW 
	
ant_world.o: ant_world.cc
	$(NVCC) $(NVCCFLAGS)  ant_world.cc

clean:
	rm -f ant_world ant_world.o
