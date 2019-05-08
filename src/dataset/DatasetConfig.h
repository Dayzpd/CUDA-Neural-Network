
#ifndef DATASET_CONFIG_H
#define DATASET_CONFIG_H

#include <string>

struct DatasetConfig
{
std::string dataset_path;
int batch_size;
int dim_size;
int feature_size;
int num_batches;
int num_classes;
int num_test_features;

DatasetConfig(std::string dataset_path);

~DatasetConfig();
};

#endif
