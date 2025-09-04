
SRC := ./tensorCoreHGemm_sharedmem.cu    \
       ../utils/cublas_lay_egg.cu     \
	   ../utils/matrix_init_print.cu


EXE := tensorCoreHGemm_sharedmem

all: $(EXE)

INC       := -I../utils
LD_FLAG   := -L/usr/local/cuda/lib64 -lcublas
ARCH_FLAG := --gpu-architecture=sm_120

$(EXE): $(SRC)
	nvcc $(ARCH_FLAG) $^ -o $@ -Xcompiler -fopenmp -lgomp $(LD_FLAG) $(INC)

.PHONY: clean
clean:
	-rm -rf $(EXE)

