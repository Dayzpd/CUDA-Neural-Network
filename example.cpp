
#include "src/dataset/DatasetConfig.h"
#include "src/dataset/Dataset.h"
#include "src/neurons/Dim.h"
#include "src/neurons/Neurons.h"
#include "src/layer/FullyConnected.h"
#include "src/layer/ReLU.h"
#include "src/output/SoftmaxCE.h"

#include <iostream>
#include <memory>
#include <string>
#include <time.h>
#include <vector>

Neurons& forward_propagate(Neurons& data, FullyConnected& fc_1, ReLU& relu_1,
  FullyConnected& fc_2, ReLU& relu_2, FullyConnected& fc_3, SoftmaxCE& sm_ce
) {
  Neurons layer_output;
  layer_output = fc_1.forward_prop(data);
  layer_output = relu_1.forward_prop(layer_output);
  layer_output = fc_2.forward_prop(layer_output);
  layer_output = relu_2.forward_prop(layer_output);
  layer_output = fc_3.forward_prop(layer_output);
  return sm_ce.forward_prop(layer_output);
}

void back_propagate(Neurons& targets, FullyConnected& fc_1, ReLU& relu_1,
  FullyConnected& fc_2, ReLU& relu_2, FullyConnected& fc_3, SoftmaxCE& sm_ce,
  float learning_rate
) {
  Neurons layer_error;
  layer_error = sm_ce.back_prop(targets);
  layer_error = fc_3.back_prop(layer_error, learning_rate);
  layer_error = relu_2.back_prop(layer_error, learning_rate);
  layer_error = fc_2.back_prop(layer_error, learning_rate);
  layer_error = relu_1.back_prop(layer_error, learning_rate);
  fc_1.back_prop(layer_error, learning_rate);
}

bool check_prediction(Neurons& prediction, Neurons& target)
{
  prediction.memcpy_device_to_host();
  target.memcpy_device_to_host();

  float max_val = -1.f;
  int max_index = -1;

  for (int x = 0; x < target.host_data.size(); x++)
  {
    if (max_val < prediction.host_data[x])
    {
      max_val = prediction.host_data[x];
      max_index = x;
    }
  }

  int target_index = -1;

  for (int x = 0; x < target.host_data.size(); x++)
  {
    if (target.host_data[x] == 1)
    {
      target_index = x;
      break;
    }
  }

  return max_index == target_index;
}

int main()
{
  srand(time(NULL));

  DatasetConfig config("/home/dirichlet/Desktop/Projects/CUDA-Neural-Network/formatted_dataset");

  Dataset dataset(config);

  FullyConnected fc_1(Dim(784, 1000), "FC_1");
  ReLU relu_1("ReLU_1");
  FullyConnected fc_2(Dim(1000, 1000), "FC_2");
  ReLU relu_2("ReLU_2");
  FullyConnected fc_3(Dim(1000, 10), "FC_3");
  SoftmaxCE sm_ce("OUTPUT");

  float learning_rate = 0.001;
  float loss;

  for (int e = 0; e <= 10000; e++)
  {
    loss = 0.0f;
    for (int b = 0; b < dataset.get_num_batches(); b++)
    {
      forward_propagate(dataset.get_batch(b), fc_1, relu_1, fc_2, relu_2, fc_3,
        sm_ce);

      loss += sm_ce.loss(dataset.get_batch_targets(b));

      back_propagate(dataset.get_batch_targets(b), fc_1, relu_1, fc_2, relu_2,
        fc_3, sm_ce, learning_rate);
    }

    if (e % 10 == 0)
    {
      int correct_counter = 0;
      for (int f = 0; f < dataset.get_num_test_features(); f++)
      {
        Neurons prediction = forward_propagate(dataset.get_test_feature(f),
          fc_1, relu_1, fc_2, relu_2, fc_3, sm_ce);

        if (check_prediction(prediction, dataset.get_test_target(f)))
        {
          correct_counter++;
        }
      }

      std::cout << "Epoch: " << e << " | Loss: " << loss << " | Accuracy: " <<
        100.f * ((float)correct_counter / dataset.get_num_test_features()) <<
        "%" << std::endl;
    }
  }

  return 0;
}
