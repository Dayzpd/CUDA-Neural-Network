
#ifndef NEURONS_H
#define NEURONS_H

#include "Dim.h"

#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

class Neurons
{
private:
  bool allocated = false;

public:
  Dim dim;

  thrust::host_vector<float> host_data;
  thrust::device_vector<float> device_data;

  Neurons(size_t x = 1, size_t y = 1);

  Neurons(Dim dim);

  ~Neurons();

  void allocate_memory();

  void allocate_memory(Dim dim);

  void allocate_memory(size_t x, size_t y);

  void deallocate_memory();

  void memcpy_host_to_device();

  void memcpy_device_to_host();

  float* get_device_pointer();
};

#endif
