
#include "src/dataset/DatasetConfig.h"
#include "src/dataset/Dataset.h"
#include "src/neurons/Dim.h"
#include "src/neurons/Neurons.h"
#include "src/layer/FullyConnected.h"
#include "src/layer/ReLU.h"
#include "src/layer/Softmax.h"
#include "src/loss/CrossEntropy.h"

#include <iostream>
#include <string>
#include <time.h>
#include <vector>

int main()
{
  using namespace cuda_net;

  srand(time(NULL));

  DatasetConfig config("D:/Projects/CUDA/CUDA-Neural-Network/formatted_dataset");

  Dataset dataset(config);

  std::vector<Layer*> layers;
  layers.push_back(new FullyConnected(Dim(784, 1000)));
  layers.push_back(new ReLU());
  layers.push_back(new FullyConnected(Dim(1000, 1000)));
  layers.push_back(new ReLU());
  layers.push_back(new FullyConnected(Dim(1000, 10)));
  layers.push_back(new Softmax());

  CrossEntropy ce;
  float learning_rate = 0.01;
  Neurons prob;
  Neurons prob_delta;

  for (int e = 0; e <= 1000; e++)
  {

    float loss = 0.0f;
    for (int b = 0; b < dataset.get_num_batches(); b++)
    {
      prob = dataset.get_batch(b);

      for (auto layer : layers)
      {
        prob = layer->forward_prop(prob);
      }

      loss += ce.calculate(prob, dataset.get_batch_targets(b));

      prob_delta.allocate_memory(prob.dim);
      Neurons error = ce.calculate_deriv(prob, dataset.get_batch_targets(b), prob_delta);

      for (auto iter = layers.rbegin(); iter != layers.rend(); iter++)
      {
        error = (*iter)->back_prop(error, learning_rate);
      }

    }

    if (e % 100 == 0)
    {
      std::cout << "Epoch: " << e << "Cost: " << loss / dataset.get_num_batches() << std::endl;
    }
  }

  return 0;
}
