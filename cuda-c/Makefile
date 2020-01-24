CXX = g++
NVCC = nvcc

CFLAGS = -O3 -g -std=c++11
CFLAGS += -L. ${REAL} -lm
LDFLAGS = ${CFLAGS}

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -L${CUDA_PATH}/lib64
LDFLAGS += -lm -arch=sm_60 -lcudart -lcusparse
INCLUDE = -I./inc
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc

BIN = ./bin
SRC = ./src
OBJ = ./obj

OBJ_SUF = .o
OS_SUF = .s.o

OBJS_LIB = $(OBJ)/nsparse.o
OBJS_SPGEMM_HASH_S = $(OBJS_LIB) $(OBJ)/kernel/kernel_spgemm_hash_s.o $(OBJ)/sample/spgemm/spgemm_hash.o

spgemm_hash :
	make spgemm_hash_s

spgemm_hash_s : $(OBJS_SPGEMM_HASH_S:$(OBJ_SUF)=$(OS_SUF))
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $(OBJS_SPGEMM_HASH_S:$(OBJ_SUF)=$(OS_SUF)) $(LDFLAGS) $(INCLUDE)

$(OBJ)/%$(OS_SUF) : $(SRC)/%.c
	mkdir -p $(dir $@)
	$(CXX) -c -DFLOAT $(CFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OS_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DFLOAT $(LDFLAGS) $(INCLUDE) -o $@ $<

clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
