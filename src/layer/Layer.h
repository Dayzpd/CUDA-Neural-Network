using namespace std;

class Layer
{
  public:
    const vector<string> ALLOWED_LAYER_TYPES = {
      "INPUT", "HIDDEN", "OUTPUT"
    };

    Layer();

    virtual Layer(string layer_type, int num_neurons);

    virtual Layer(string layer_type);

    virtual void populate_layer(int num_neurons) = 0;

    virutal bool connect_layer(Layer next_layer) = 0;

    virtual void get_num_neurons();

    virtual void set_num_neurons(int num_neurons);

    virtual void get_layer_type();

    virtual void set_layer_type(string layer_type);
}
