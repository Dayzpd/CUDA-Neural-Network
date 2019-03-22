
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
  /// <param name="loss_type">The <c>LossFactory</c> class supplies constants
  /// that must to be used (e.g. <c>LossFactory::BINARY_CROSS_ENTROPY</c>,
  /// <c>LossFactory::HINGE</c>, <c>LossFactory::MEAN_SQUARED_ERROR</c>,
  /// <c>LossFactory::MULTI_CLASS_CROSS_ENTROPY</c></param name>
  /// <returns>Returns a pointer to an loss function of the specified
  /// type.</returns>
  LossFunction& LossFactory::create(Type loss_type)
  {
    switch (loss_type)
    {
      case BINARY_CROSS_ENTROPY:
        static BinaryCrossEntropy bce_instance;
        return bce_instance;
      case HINGE:
        static Hinge hinge_instance;
        return hinge_instance;
      case MEAN_SQUARED_ERROR:
        static MeanSquaredError mse_instance;
        return mse_instance;
      case MULTI_CLASS_CROSS_ENTROPY:
        static MultiClassCrossEntropy mc_ce_instance;
        return multi_class_ce_instance;
    }
    throw runtime_error("An invalid loss function type was given. " +
      "Accepted types include: " +
      "LossFactory::BINARY_CROSS_ENTROPY, " +
      "LossFactory::HINGE, " +
      "LossFactory::MEAN_SQUARED_ERROR, " +
      "LossFactory::MULTI_CLASS_CROSS_ENTROPY");
  }

}
