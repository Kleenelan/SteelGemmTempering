

EXE := cublasHGemm_main

all: $(EXE)

%: %.cu
	nvcc --gpu-architecture=sm_120 $< -o $@ -L/usr/local/cuda/lib64 -lcublas -I../utils


.PHONY: clean
clean:
	-rm -rf $(EXE)

