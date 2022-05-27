# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall -ftree-vectorize -march=native
PAR_FLAG = -fopenmp
GENERATE = -D GENERATE
#GENERATE = 
TIMING = -D TIMING

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

APPS = matadd transpose

.PHONY: all

all: $(APPS)

matadd: matadd.c mmio.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

transpose: transpose.c mmio.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS)
