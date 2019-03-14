# CUDA Neural Nets
## Brief Explanation (and Disclaimer!)
Currently, I am working on implementing a framework for training neural nets in
C++ with CUDA. If for whatever reason anyone stumbles upon this, I would not
recommend that you integrate this into your project. The primary purpose of this
is for my own educational gratification, and you would be far better off using
Tensorflow, PyTorch, etc. However, if you happen to share my interest in
learning how neural nets are implemented from scratch, feel free to look at the
code, take whatever you would like, and if you feel that it's absolute trash,
leave a comment! At the time of writing, my experience with C++ and CUDA is
limited so my knowledge possesses a vast multiplicity of blind spots.

## Architecture Plan
The following documents the current plan of how I am developing the framework.
As previously mentioned, my experience is limited so as I discover new things or
run into problems this may change slightly.

### Components
- Network (inherits from doubly linked list)
- Layers
  - Input Layer (in progress)
  - Fully Connected Layer (in progress)
  - Output Layer (in progress)
- Neurons (inherits from matrix)
- Activation Function
  - Fast Sigmoid (Approximates Sigmoid function)
  - Leaky ReLU
  - ReLU
  - TanH
- Loss Function (undetermined)

### Putting the Puzzle Pieces Together
A network needs and an input layer, output layer, and whatever hidden layers
that the user specifies (Fully Connected, Convolutional, Pooling). Layers are
added to the network using a [factory method](https://en.wikipedia.org/wiki/Factory_method_pattern) inside the Network class. In order to build any given
layer, you must supply it with the numbers of neurons it should have, and the
activation function it should use (activation function is not applicable for
input layers). Layers must be added to the network in proper order (i.e. input
layer -> hidden layer(s) -> output layer).

As for selecting activation functions, this seems to me to be a case wherein
static polymorphism rears its head. Since the


### Flow of Network Creation
