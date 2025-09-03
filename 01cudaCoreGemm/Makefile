EXE := cudaCoreGemm
# hello_gemm.fp16.cudaCore.v01
#EXE := cudaTensorCoreGemm
all: $(EXE)

%: %.cu
	nvcc  --gpu-architecture=sm_120 $< -o $@ -Icuda-samples/Common

.PHONY: clean
clean:
	-rm -rf $(EXE)