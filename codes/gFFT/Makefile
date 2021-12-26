NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	    = gFFT
OBJ	    = gFFT.o

default: $(EXE)

gFFT.o: gFFT.cu
	$(NVCC) -c -o $@ gFFT.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
