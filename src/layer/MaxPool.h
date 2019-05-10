
#ifndef MAX_POOL_H
#define MAX_POOL_H

#include "Layer.h"

#include <thrust/device_vector.h>

class MaxPool : public Layer
{
private:
  Neurons input;
  Neurons max_indices;
  Neurons output;
  Neurons delta;

  size_t kernel_dim;
  std::string name;
  bool verbose;

public:
  MaxPool(size_t kernel_dim, std::string name, bool verbose = false);

  ~MaxPool();

  Neurons& get_max_indices();

  Neurons& forward_prop(Neurons& input);

  Neurons& back_prop(Neurons& input, float learning_rate);
};

#endif
