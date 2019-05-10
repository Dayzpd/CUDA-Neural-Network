
#ifndef CONV_2D_H
#define CONV_2D_H

#include "Layer.h"
#include "../neurons/Neurons.h"

#include <string>

class Conv2D : public Layer
{
  private:
    Neurons input;
    Neurons kernel;
    float bias;
    Neurons output;
    Neurons delta;

    size_t kernel_dim;
    std::string name;
    bool verbose;

  public:
    Conv2D(size_t kernel_dim, std::string name, bool verbose = false);

    ~Conv2D();

    void init_kernel();

    Neurons& forward_prop(Neurons& input);

    Neurons& back_prop(Neurons& error, float learning_rate);

    void backprop_error(Neurons& error);

    void update_kernel(Neurons& error, float learning_rate);

    void update_bias(Neurons& error, float learning_rate);
};

#endif
