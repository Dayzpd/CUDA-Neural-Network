
#ifndef NEURONS_H
#define NEURONS_H

#include "Dim.h"

#include <memory>

namespace neural_network
{

  class Neurons
  {
    private:
      bool is_host_reserved;
      bool is_device_reserved;

      void reserve_host_memory();

      void reserve_device_memory();

    public:
      Dim dim;

      std::shared_ptr<float> host_neurons;
      std::shared_ptr<float> device_neurons;

      ~Neurons();

      Neurons(size_t x = 1, size_t y = 1);

      Neurons(Dim dim);

      void reserve_memory();

      void reserve_memory(Dim dim);

      void memcpy_host_to_device();

      void memcpy_device_to_host();

      float& operator[](const int index);

	    const float& operator[](const int index) const;
  }

}

#endif
