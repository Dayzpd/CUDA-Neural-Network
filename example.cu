#include "batches/MiniBatch.h"
#include "network/Network.h"

#include <string>

int main()
{
  std::string data_path = "D:/Projects/CUDA/CUDA-Neural-Network/dataset/";

  MiniBatch batches = new MiniBatch(50, 784, 10);
  batches.register_classes(data_path + "object_labels.csv");
  batches.load_train(data_path + "train_labels.csv", data_path + "train/");
  batches.load_test(data_path + "test_labels.csv", data_path + "test/");

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
      prob = network.forward_propagate(batches.at(b));
      network.back_propagate(prob, batches.at(b), learning_rate);
      float loss += network.loss_func.calculate(output, actuals.at(b));
    }

    if (e % 100 == 0)
    {
      printf("Epoch: %d, Cost: %f", e, loss / batches.get_num_batches());
    }
  }
}
