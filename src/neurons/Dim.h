
#ifndef DIM_H
#define DIM_H

namespace neural_network
{

  struct Dim
  {
      size_t x, y, z;

      Dim(size_t x = 1, size_t y = 1, size_t z = 1);
  }

}

#endif
