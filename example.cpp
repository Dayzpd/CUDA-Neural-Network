#include "src/dataset/DatasetConfig.h"
#include "src/dataset/Dataset.h"
#include "network/Network.h"

#include <string>

int main()
{
  DatasetConfig config("D:/Projects/CUDA/CUDA-Neural-Network/formatted_dataset");

  Dataset dataset = new Dataset(config);

  Network network = new Network("CROSS_ENTROPY");
  network.add_layer("FULLY_CONNECTED", 784, 1000);
  network.add_layer("RELU");
  network.add_layer("FULLY_CONNECTED", 1000, 1000);
  network.add_layer("RELU");
  network.add_layer("FULLY_CONNECTED", 1000, 10);
  network.add_layer("SOFTMAX");

  for (int e = 0; e <= 1000; e++)
  {
    float loss = 0.0f;
    for (int b = 0; b < batches.get_num_batches(); b++)
    {
      prob = network.forward_propagate(dataset.get_batch(b));
      network.back_propagate(prob, dataset.get_batch_targets(b), learning_rate);
      float loss += network.loss_func.calculate(output,
        dataset.get_batch_targets(b));
    }

    if (e % 100 == 0)
    {
      printf("Epoch: %d, Cost: %f", e, loss / batches.get_num_batches());
    }
  }
}
