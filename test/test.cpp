
#include "../src/dataset/DatasetConfig.h"
#include "../src/dataset/Dataset.h"
#include "../src/layer/FullyConnected.h"
#include "../src/layer/ReLU.h"
#include "../src/layer/Softmax.h"
#include "../src/loss/CrossEntropy.h"
#include "../src/neurons/Dim.h"
#include "../src/neurons/Neurons.h"

#include <iostream>
#include <string>
#include <vector>

/*
  The following is a visual test ensuring that the input is being read
  correctly. Actual normalized pixel values aren't shown, but rather if the
  value is above 0, a "#" is displayed to the terminal. Else, a " " is
  displayed. This results in an ASCII art number which is easier to confirm
  whether or not the image read from the csv file is indeed the correct class
  that the target stores.
*/
void dataset_test()
{
  DatasetConfig config("../formatted_dataset");

  Dataset dataset(config);

  Neurons batch = dataset.get_batch(0);
  Neurons targets = dataset.get_batch_targets(0);
  batch.memcpy_device_to_host();
  targets.memcpy_device_to_host();

  // Picking three indexes for testing
  std::vector<int> test_indices = {0, 13, 24};

  std::cout <<  "### BATCH TEST ###" << std::endl;
  std::cout <<  "###### Size Check ######" << std::endl;

  // Batch size is 25 and MNIST image length is 784 and num classes is 10
  std::cout <<  "Batch Size: (" << batch.dim.x << ", " << batch.dim.y << ")" << std::endl;

  if (batch.dim.x == 25 && batch.dim.y == 784)
  {
    std::cout <<  "Batch Size: PASS" << std::endl;
  }
  else
  {
    std::cerr <<  "Batch Size: FAIL" << std::endl;
  }

  std::cout <<  "Targets Size: (" << targets.dim.x << ", " << targets.dim.y << ")" << std::endl;

  if (targets.dim.x == 25 && targets.dim.y == 10)
  {
    std::cout <<  "Targets Size: PASS" << std::endl;
  }
  else
  {
    std::cerr <<  "Targets Size: FAIL" << std::endl;
  }

  std::cout <<  "###### Content Check ######" << std::endl;

  for (int i = 0; i < test_indices.size(); i++)
  {
    std::cout <<  "######### Index: " << test_indices[i] << " #########" << std::endl;
    std::cout <<  "######### Feature #########" << std::endl;

    // Print feature pixels from batch
    for (int x = 0; x < 28; x++)
    {
      for (int y = 0; y < 28; y++)
      {
        int pixel_num = test_indices[i] * 784 + x * 28 + y;
        if (batch.host_data[pixel_num] > 0)
        {
          printf("#");
        }
        else
        {
          printf(" ");
        }
      }
      printf("\n");
    }

    std::cout <<  "######### Target #########" << std::endl;

    // Print target for test index
    for (int x = 0; x < 10; x++)
    {
      int pixel_num = test_indices[i] * 10 + x;
      if (targets.host_data[pixel_num] == 1)
      {
        printf("class: %d", x);
      }
    }
    printf("\n");
  }
}

void fc_test()
{
  std::vector<float> vals = {
    0.0, 0.0, 0.65, 0.43, 0.92,
    0.87, 0.0, 0.34, 0.0, 0.32,
    0.54, 0.58, 0.0, 0.0, 0.66,
    0.0, 0.12, 0.0, 0.43, 0.98
  };

  Neurons test_input(4, 5);
  test_input.allocate_memory();
  thrust::copy(vals.begin(), vals.end(), test_input.device_data.begin());

  FullyConnected fc_test(Dim(5, 4), "FC_TEST", true);
  Neurons fc_test_output = fc_test.forward_prop(test_input);
}

void relu_test()
{
  std::vector<float> vals = {
    -0.2, 0.33, -0.65, -0.43, 0.92,
    0.87, 0.05, 0.34, 0.01, 0.32,
    0.54, 0.58, -0.88, 0.24, 0.66,
    -0.77, 0.12, -0.65, 0.43, 0.98
  };

  std::vector<float> check = {
    0, 0.33, 0, 0, 0.92,
    0.87, 0.05, 0.34, 0.01, 0.32,
    0.54, 0.58, 0, 0.24, 0.66,
    0, 0.12, 0, 0.43, 0.98
  };

  Neurons test_input(4, 5);
  test_input.allocate_memory();
  thrust::copy(vals.begin(), vals.end(), test_input.device_data.begin());

  ReLU relu_test("RELU_TEST");
  Neurons relu_test_output = relu_test.forward_prop(test_input);

  relu_test_output.memcpy_device_to_host();

  for (int x = 0; x < 20; x++)
  {
    if (relu_test_output.host_data[x] != check[x])
    {
      std::cout << "FAIL\n" << std::endl;
      return;
    }
    else
    {
      std::cout << relu_test_output.host_data[x] << " = " << check[x] << std::endl;
    }
  }

  std::cout << "PASS\n" << std::endl;
}

void softmax_ce_test()
{
  std::vector<float> vals = {
    -0.2, 0.33, -0.65, -0.43,
    0.87, 0.05, 0.34, 0.01,
    0.54, 0.58, -0.88, 0.24
  };

  std::vector<float> targets = {
    0, 1, 0, 0,
    0, 0, 1, 0,
    1, 0, 0, 0
  };

  Neurons test_input(3, 4);
  test_input.allocate_memory();
  thrust::copy(vals.begin(), vals.end(), test_input.device_data.begin());

  Neurons test_actuals(3, 4);
  test_actuals.allocate_memory();
  thrust::copy(targets.begin(), targets.end(), test_actuals.device_data.begin());

  Softmax softmax_test("SOFTMAX_TEST", true);
  Neurons softmax_test_output = softmax_test.forward_prop(test_input);
  softmax_test_output.memcpy_device_to_host();

  for (int x = 0; x < 12; x++)
  {
    std::cout << "SOFTMAX[" << x << "] = " << softmax_test_output.host_data[x] << std::endl;
  }

  //CrossEntropy ce_test;
  float loss = softmax_test.loss(test_actuals);

  std::cout << "LOSS = " << loss << std::endl;

  Neurons test_error;
  test_error = softmax_test.back_prop(test_actuals);
  test_error.memcpy_device_to_host();
  /*
  for (int x = 0; x < test_error.dim.x * test_error.dim.y; x++)
  {
    std::cout << "CE_ERR[" << x << "] = " << test_error.host_data[x] << std::endl;
  }

  test_error = softmax_test.back_prop(test_error, .01);

  test_error.memcpy_device_to_host();
*/
  for (int x = 0; x < test_error.dim.x * test_error.dim.y; x++)
  {
    std::cout << "SOFTMAX_ERR[" << x << "] = " << test_error.host_data[x] << std::endl;
  }
}


int main()
{
  std::cout << "######### Test Menu #########" << std::endl;
  std::cout << "Option 0: dataset test" << std::endl;
  std::cout << "Option 1: fc layer test" << std::endl;
  std::cout << "Option 2: relu layer test" << std::endl;
  std::cout << "Option 3: softmax ce test" << std::endl;
  std::cout << "Selection: ";

  int option;
  std::cin >> option;

  switch (option)
  {
    case 0:
      dataset_test();
      break;
    case 1:
      fc_test();
      break;
    case 2:
      relu_test();
      break;
    case 3:
      softmax_ce_test();
      break;
  }

  return 0;
}
