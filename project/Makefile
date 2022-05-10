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

APPS = fft

.PHONY: all

all: $(APPS)

fft.o:		fft.cu
		$(NVCC) $(CU_FLAGS) $(GENERATE) $(TIMING)  -c $< -o $@ $(LFLAGS)

fft.ptx:		fft.cu
		$(NVCC) $(CU_FLAGS) $(GENERATE) $(TIMING)  --ptx $< -o $@ $(LFLAGS)

fft_cpu.o:		fft_cpu.cpp
		$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING)  -c $< -o $@ $(LFLAGS)

fft: fft.o fft_cpu.o
	$(NVCC) $(CU_FLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -f $(APPS) *.o
