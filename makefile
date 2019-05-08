CUDA_ROOT_DIR=/usr/local/cuda-9.0

CC=gcc
CC_FLAGS=-std=c++11
CC_LIBS=

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch=sm_61 -std=c++11 -lineinfo
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcublas

##########################################################

## Project file structure ##

# Source file directory:
DATASET_DIR = src/dataset
LAYER_DIR = src/layer
OUTPUT_DIR = src/output
NEURONS_DIR = src/neurons

# Object file directory:
OBJ_DIR = bin

##########################################################

## Make variables ##

# Target executable name:
EXE = example

# Object files:
OBJS = $(OBJ_DIR)/example.o $(OBJ_DIR)/Dataset.o $(OBJ_DIR)/DatasetConfig.o $(OBJ_DIR)/FullyConnected.o $(OBJ_DIR)/ReLU.o $(OBJ_DIR)/SoftmaxCE.o $(OBJ_DIR)/Dim.o $(OBJ_DIR)/Neurons.o

##########################################################

## Compile ##


# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/example.o : example.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile Dataset files:
$(OBJ_DIR)/%.o : $(DATASET_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile Layer files:
$(OBJ_DIR)/%.o : $(LAYER_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile Loss files:
$(OBJ_DIR)/%.o : $(OUTPUT_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile Neurons files:
$(OBJ_DIR)/%.o : $(NEURONS_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean objects in object directory.
clean:
	$(RM) bin/*.o
