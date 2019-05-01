
#ifndef DIM_H
#define DIM_H

namespace cuda_net
{

  struct Dim
  {
      size_t x, y, z;

      Dim(size_t x = 1, size_t y = 1, size_t z = 1);
  };

}

#endif
