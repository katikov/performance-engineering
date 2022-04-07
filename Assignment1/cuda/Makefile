# See LICENSE.txt for license details.
NVCC = nvcc
CXX_FLAGS += -std=c++11 -O3 -Wall -ftree-vectorize 
CU_FLAGS += -O3 -g --ptxas-options=-v
PAR_FLAG = -fopenmp
#GENERATE = -D GENERATE
GENERATE = 
TIMING = -D TIMING

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

APPS = matmul transpose

.PHONY: all

all: $(APPS)

matmul.o:		matmul.cu
		$(NVCC) $(CU_FLAGS) $(GENERATE) $(TIMING)  -c $< -o $@ $(LFLAGS)

matmul.ptx:		matmul.cu
		$(NVCC) $(CU_FLAGS) $(GENERATE) $(TIMING)  --ptx $< -o $@ $(LFLAGS)

mmio.o:		mmio.c
		$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING)  -c $< -o $@ $(LFLAGS)

matmul: matmul.o mmio.o
	$(NVCC) $(CU_FLAGS) $^ -o $@

transpose: transpose.c mmio.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS) *.o
