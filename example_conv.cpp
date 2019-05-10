
#include "src/dataset/DatasetConfig.h"
#include "src/dataset/Dataset.h"
#include "src/neurons/Dim.h"
#include "src/neurons/Neurons.h"
#include "src/layer/FullyConnected.h"
#include "src/layer/ReLU.h"
#include "src/layer/Conv2D.h"
#include "src/layer/MaxPool.h"
#include "src/output/SoftmaxCE.h"

#include <iostream>
#include <memory>
#include <string>
#include <time.h>
#include <vector>

Neurons& forward_propagate(Neurons& data, Conv2D& conv_1, ReLU& relu_1,
  Conv2D& conv_2, ReLU& relu_2, FullyConnected& fc_1, SoftmaxCE& sm_ce
) {
  Neurons layer_output;
  layer_output = conv_1.forward_prop(data);
  layer_output = relu_1.forward_prop(layer_output);
  layer_output = conv_2.forward_prop(layer_output);
  layer_output = relu_2.forward_prop(layer_output);
  layer_output = fc_1.forward_prop(layer_output);
  return sm_ce.forward_prop(layer_output);
}

void back_propagate(Neurons& targets, Conv2D& conv_1, ReLU& relu_1,
  Conv2D& conv_2, ReLU& relu_2, FullyConnected& fc_1, SoftmaxCE& sm_ce,
  float learning_rate
) {
  Neurons layer_error;
  layer_error = sm_ce.back_prop(targets);
  layer_error = fc_1.back_prop(layer_error, learning_rate);
  layer_error = relu_2.back_prop(layer_error, learning_rate);
  layer_error = conv_2.back_prop(layer_error, learning_rate);
  layer_error = relu_1.back_prop(layer_error, learning_rate);
  conv_1.back_prop(layer_error, learning_rate);
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

  Conv2D conv_1(6, "CONV_1");
  ReLU relu_1("ReLU_1");
  Conv2D conv_2(6, "CONV_2");
  ReLU relu_2("ReLU_2");
  FullyConnected fc_1(Dim(324, 10), "FC_1");
  SoftmaxCE sm_ce("OUTPUT");

  float learning_rate = 0.001;
  float loss;

  for (int e = 0; e <= 10000; e++)
  {
    loss = 0.0f;
    for (int b = 0; b < dataset.get_num_batches(); b++)
    {
      forward_propagate(dataset.get_batch(b), conv_1, relu_1, conv_2, relu_2,
        fc_1, sm_ce);

      loss += sm_ce.loss(dataset.get_batch_targets(b));

      back_propagate(dataset.get_batch_targets(b), conv_1, relu_1, conv_2,
        relu_2, fc_1, sm_ce, learning_rate);
    }

    if (e % 10 == 0)
    {
      int correct_counter = 0;
      for (int f = 0; f < dataset.get_num_test_features(); f++)
      {
        Neurons prediction = forward_propagate(dataset.get_test_feature(f),
          conv_1, relu_1, conv_2, relu_2, fc_1, sm_ce);

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
