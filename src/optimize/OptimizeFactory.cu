#include "ConnectionFunction.h"

#include <memory>
#include <stdexcept>

/// <summary><c>create</c> serves as a factory function to create
/// strategies for optimization functions.</summary>
/// <param name="layer_type">The <c>OptimizationFactory</c> class supplies
/// constants that must to be used (e.g. <c>OptimizationFactory::GRAD_DESC</c>).
/// </param name>
/// <returns>Returns a pointer to the optimization strategy of the specified
/// type.</returns>
std::unique_ptr<OptimizationFunction> OptimizationFactory::create(
  Type optimize_type
) {
  switch (optimize_type)
  {
    case GRAD_DESC:
      return std::make_unique<GradientDescent>();
  }
  throw runtime_error(layer_type + " is an invalid optimization type. " +
    "Accepted types include: " +
    "OptimizationFactory::GRAD_DESC.");
}
