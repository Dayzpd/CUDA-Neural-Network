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
- Network
  - Actions:
    - Constructs layers of Neurons
    - Trains network once network layout is constructed
  - Supply these for instantiation:
    - Input size
    - Learning rate
    - Output classes
    - Loss Function strategy (e.g. Binary Cross Entropy, Hinge, Mean Squared,
      Multi Class Cross Entropy)
    - Optimize Function strategy (e.g. Gradient Descent)
  - Before training, construct layers in order of how you would like
    - Layers, from the perspective of the network, are Neurons objects
    - Neurons
      - the numbers of neurons it should have
      - activation function strategy (e.g. Fast Sigmoid, Leaky ReLU, ReLU, TanH)
      - layer strategy (e.g. Convolutional, Fully Connected, Pooling, Output)
