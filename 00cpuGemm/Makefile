

EXE := cpuHGemm

all: $(EXE)

%: %.cu
	nvcc $< -o $@ -Xcompiler -fopenmp -lgomp -L/usr/local/cuda/lib64 -lcublas -I..


.PHONY: clean
clean:
	-rm -rf $(EXE)

