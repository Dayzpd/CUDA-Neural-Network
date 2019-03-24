# CUDA Neural Nets
## Brief Explanation (and Disclaimer!)
Currently, I am implementing a framework for training neural nets in
C++ with CUDA. If for whatever reason anyone stumbles upon this, I would not
recommend that you integrate this into your project. The primary purpose of this
is for my own educational gratification, and you would be far better off using
Tensorflow, PyTorch, etc. However, if you happen to share my interest in
learning how neural nets are implemented from scratch, feel free to look at the
code, take whatever you would like, or if you feel that it's absolute trash,
leave a comment.

## Architecture Plan
The following documents the current plan of how I am developing the framework.
As previously mentioned, my experience is limited so as I discover new things or
run into problems this may change slightly.

- Layers
  - Can be created using the Layer class which constructs and returns a Neurons
    object. Various activation functions are available such Fast Sigmoid,
    Leaky ReLU, ReLU, and TanH. CRTP pattern is used for activation functions
    and they are created using a factory.
- Connections
  - In between adjacent Neurons objects, a Connection object determines how
    individual neurons in one layer connect to neurons in the next layer (e.g.
    Convolutional, Fully Connected, Pooling, Output).
- Training (TBD)
