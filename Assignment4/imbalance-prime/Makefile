# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall -ftree-vectorize -march=native
PAR_FLAG = -fopenmp
TIMING = -D TIMING

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

APPS = prime_before prime_after 

.PHONY: all

all: $(APPS)

prime_before: prime_before.c
	$(CXX) $(CXX_FLAGS)  -o $@ $^ $(LFLAGS)

prime_after: prime_after.c
	$(CXX) $(CXX_FLAGS)  -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS)
