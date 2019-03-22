
#include "BinaryCrossEntropy.h"
#include "Hinge.h"
#include "MeanSquaredError.h"
#include "MultiClassCrossEntropy.h"

#include <memory>
#include <stdexcept>

namespace neural_network
{
  /// <summary><c>create</c> serves as a factory function that returns an
  /// loss function of the specified type.</summary>
  /// <param name="loss_type">The <c>LossFactory</c>
  /// class supplies constants that must to be used (e.g.
  /// <c>LossFactory::BINARY_CROSS_ENTROPY</c>,
  /// <c>LossFactory::HINGE</c>,
  /// <c>LossFactory::MEAN_SQUARED_ERROR</c>,
  /// <c>LossFactory::MULTI_CLASS_CROSS_ENTROPY</c></param name>
  /// <returns>Returns a pointer to an loss function of the specified
  /// type.</returns>
  std::unique_ptr<LossFunction> LossFactory::create(Type loss_type)
  {
    switch (loss_type)
    {
      case BINARY_CROSS_ENTROPY:
        return std::make_unique<BinaryCrossEntropy>();
      case HINGE:
        return std::make_unique<Hinge>();
      case MEAN_SQUARED_ERROR:
        return std::make_unique<MeanSquaredError>();
      case MULTI_CLASS_CROSS_ENTROPY:
        return std::make_unique<MultiClassCrossEntropy>();
    }
    throw runtime_error("An invalid loss function type was given. " +
      "Accepted types include: " +
      "LossFactory::BINARY_CROSS_ENTROPY, " +
      "LossFactory::HINGE, " +
      "LossFactory::MEAN_SQUARED_ERROR, " +
      "LossFactory::MULTI_CLASS_CROSS_ENTROPY");
  }

}
