
#include "LossFactory.h"
#include "CrossEntropy.h"

#include <memory>
#include <stdexcept>

namespace cuda_net
{
  const std::string CROSS_ENTROPY = "CROSS_ENTROPY";

  LossFactory& LossFactory::get_instance()
  {
    static LossFactory factory_instance;
    return factory_instance;
  }

  /// <summary><c>create</c> serves as a factory function that returns an
  /// loss function of the specified type.</summary>
  /// <param name="loss_type">The <c>LossFactory</c> class supplies constants
  /// that must to be used (e.g. <c>LossFactory::CROSS_ENTROPY</c>).
  /// </param name>
  /// <returns>Returns a pointer to the loss function of the specified
  /// type.</returns>
  std::unique_ptr<LossFunction> LossFactory::create(std::string loss_type)
  {
    switch (loss_type)
    {
      case CROSS_ENTROPY:
        return std::make_unique<CrossEntropy>();
    }
    throw runtime_error("An invalid loss function type was given. " +
      "Accepted types include: " +
      "LossFactory::CROSS_ENTROPY");
  }

}
