target    := nmf
sources   := matrix.cu nmf.cu

opt  ?= -O3 --ptxas-options=--warn-on-double-precision-use
dbg  ?=
warn ?= -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wno-unused-parameter
arch := -arch=sm_62

cudaroot  := /usr/local/cuda

nvcc        := $(cudaroot)/bin/nvcc
cuflags     += $(arch) $(opt) $(dbg) $(warn)
lib_paths := -L$(cudaroot)/lib64
inc_paths := -I$(cudaroot)/include
libraries := -lcuda -lcudart -lcublas

objects := $(patsubst %.cu,%.o,$(sources))


$(target): $(objects)
	$(nvcc) -o $@ $(lib_paths) $(libraries) $(objects)

%.o: %.cu
	$(nvcc) -c $^ $(cuflags) $(inc_paths) -o $@

clean:
	rm -f *.o $(target) makefile.*dep *.linkinfo

