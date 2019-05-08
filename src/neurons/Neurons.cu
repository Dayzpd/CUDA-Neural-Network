#include "Neurons.h"

#include <iostream>

#include <thrust/system_error.h>

Neurons::Neurons(size_t x, size_t y) : dim(x, y), allocated(false)
{

}

Neurons::Neurons(Dim dim) : Neurons(dim.x, dim.y)
{

}

Neurons::~Neurons()
{
  device_data.clear();
  device_data.shrink_to_fit();
  host_data.clear();
  host_data.shrink_to_fit();
}

// Concrete allocation primarily used for weights and biases that have fixed
// user-defined sizes.
void Neurons::allocate_memory()
{
  if (!allocated)
  {
    try
    {
      host_data = thrust::host_vector<float>(dim.x * dim.y);
      device_data = thrust::device_vector<float>(dim.x * dim.y);
    }
    catch(thrust::system_error &e)
    {
      std::cerr << "Error [Neurons::allocate_memory()]: Failed to \
        allocate memory." << std::endl;
      exit(-1);
    }

    allocated = true;
  }
}

// On the fly allocation primarily used to allocate layer outputs.
void Neurons::allocate_memory(Dim dim)
{
  if (!allocated)
  {
    this->dim = dim;

    try
    {
      host_data = thrust::host_vector<float>(dim.x * dim.y, 0.0f);
      device_data = host_data;
    }
    catch(thrust::system_error &e)
    {
      std::cerr << "Error [Neurons::allocate_memory(Dim dim)]: Failed to \
        allocate memory." << std::endl;
      exit(-1);
    }

    allocated = true;
  }
}

void Neurons::allocate_memory(size_t x, size_t y)
{
  allocate_memory(Dim(x, y));
}

void Neurons::memcpy_host_to_device()
{
  device_data = host_data;
}

void Neurons::memcpy_device_to_host()
{
  host_data = device_data;
}

float* Neurons::get_device_pointer()
{
  return thrust::raw_pointer_cast(device_data.data());
}
