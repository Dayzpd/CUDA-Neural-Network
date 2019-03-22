#include "ConnectionFactory.h"

#include <memory>
#include <stdexcept>

/// <summary><c>create</c> serves as a factory function to establish
/// strategies for connecting neurons in a given layer to neurons in a
/// previous Layer.</summary>
/// <param name="connection_type">The <c>ConnectionFactory</c> class supplies
/// constants that must to be used (e.g. <c>ConnectionFactory::AVG_POOL</c>,
/// <c>ConnectionFactory::CONV</c>, <c>ConnectionFactory::MAX_POOL</c>,
/// <c>ConnectionFactory::FC</c>, or <c>ConnectionFactory::OUTPUT</c>).
/// </param name>
/// <returns>Returns a pointer to the connection strategy of the specified type.
///</returns>
std::unique_ptr<Connection> ConnectionFactory::create(Type connection_type)
{
  switch (connection_type)
  {
    case AVG_POOL:
      return std::make_unique<AvgPool>();
    case CONV:
      return std::make_unique<Conv>();
    case MAX_POOL:
      return std::make_unique<MaxPool>();
    case FC:
      return std::make_unique<FullyConnected>();
    case OUTPUT:
      return std::make_unique<Output>();
  }
  throw runtime_error(layer_type + " is an invalid connection type. " +
    "Accepted types include: " +
    "ConnectionFactory::AVG_POOL, " +
    "ConnectionFactory::CONV, " +
    "ConnectionFactory::MAX_POOL, " +
    "ConnectionFactory::FC, " +
    "ConnectionFactory::OUTPUT");
}
