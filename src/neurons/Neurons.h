
#ifndef NEURONS_H
#define NEURONS_H

#include "Dim.h"

#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace neural_network
{

  class Neurons
  {
    public:
      Dim dim;

      thrust::host_vector<float> host_data;
      thrust::device_vector<float> device_data;

      ~Neurons();

      Neurons(size_t x = 1, size_t y = 1, size_t z = 1);

      Neurons(Dim dim);

      void memcpy_host_to_device();

      void memcpy_device_to_host();

      float* get_device_neurons();
  }

}

#endif
