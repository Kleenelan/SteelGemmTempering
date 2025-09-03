

EXE := cublasHGemm

all: $(EXE)

%: %.cu
	nvcc $< -o $@ -L/usr/local/cuda/lib64 -lcublas -I..


.PHONY: clean
clean:
	-rm -rf $(EXE)

