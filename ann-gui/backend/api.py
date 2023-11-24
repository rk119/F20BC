import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


layers = []
loss = 'binary'
loss_function = None
x_norm = np.empty((0,0))
y = np.empty((0,0))

app = Flask(__name__)
CORS(app)

def normalize(data):
  scaler = MinMaxScaler()
  return scaler.fit_transform(data)

def seperateData(data):
    # Split into features (X) and label (Y), convert to numpy
    X = data.iloc[:, :-1].to_numpy()
    Y = data.iloc[:, -1].to_numpy()

    return (X, Y)

### Activation Functions

class ActivationFunctions:
  def evaluate(self,x):
    pass
  def derivate(self,x):
    pass

class Identity:
  def evaluate(self,x):
    return x
  def derivative(self,x):
    return 1

class Sigmoid(ActivationFunctions):
  def evaluate(self,x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))
  def derivative(self,x):
    f = self.evaluate(x)
    return f * (1-f)

class Tanh(ActivationFunctions):
  def evaluate(self,x):
    return np.tanh(x)
  def derivative(self,x):
    f = self.evaluate(x)
    return 1 - f ** 2

class relu(ActivationFunctions):
  def evaluate(self,x):
    return np.maximum(0, x)
  def derivative(self,x):
    return (x > 0).astype(float)

### Loss Functions

class LossFunctions:
  def evaluate(self,x):
    pass
  def derivate(self,x):
    pass

# y is predictions made by the neural network
# t is target/actual numbers corresponding to inputs
class Mse(LossFunctions):
  def evaluate(self, y, t):
    return ((t - y) ** 2).mean()
  def derivative(self, y, t):
    return 2 * (y - t) / len(y)

class BinaryCrossEntropy(LossFunctions):
  def evaluate(self, y, t):
    y_pred = np.clip(y, 1e-7, 1 - 1e-7)
    term0 = (1 - t) * np.log(1 - y_pred + 1e-7)
    term1 = t * np.log(y_pred + 1e-7)
    return - (term0 + term1).mean()

  def derivative(self, y, t):
    y_pred = np.clip(y, 1e-7, 1 - 1e-7)
    return (t / y_pred) - (1 - t) / (1 - y_pred)

class Hinge(LossFunctions):
  def evaluate(self, y, t):
    return np.maximum(0, 1 - t * y).mean()

  def derivative(self, y, t):
    return np.where(t * y < 1, -t, 0)

### Neural Network

class InputLayer:
    def __init__(self, input_size):
        self.nb_nodes = input_size

    def forward(self, input_data):
        return input_data

class Layer:
    def __init__(self, input_size, nodes, activation, weights=None, biases=None):
        self.nb_nodes = nodes
        self.W = weights if weights is not None else np.random.randn(input_size, nodes)
        self.B = biases if biases is not None else np.random.randn(nodes)
        self.activation = activation

    def forward(self, input_data):
        self.X_in = input_data
        z = np.dot(input_data, self.W) + self.B
        out = self.activation.evaluate(z)
        return out

class NeuralNetwork:
    def __init__(self, configuration, position=None):
        self.layers = []
        input_size = configuration[0]

        # The input layer is simply added as a pass-through layer
        self.layers.append(InputLayer(input_size))

        # If a position vector is provided, it contains weights and biases for each layer
        if position is not None:
            for idx, (nodes, activation) in enumerate(configuration[1:]):
                weights, biases = position[idx]
                # print("layer ", idx, "W = ",weights, "B = ", biases )
                layer = Layer(input_size, nodes, activation, weights=weights, biases=biases)
                self.add(layer)
                input_size = nodes  # Update input size for the next layer
        else:
            # If no position vector, initialize layers with random weights and biases
            for nodes, activation in configuration[1:]:
                layer = Layer(input_size, nodes, activation)
                self.add(layer)
                input_size = nodes  # Update input size for the next layer

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def print_layers(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, InputLayer):
                print(f"Layer {i}: Input Layer with {layer.nb_nodes} nodes")
            elif isinstance(layer, Layer):
                print(f"Layer {i}: Hidden Layer with {layer.nb_nodes} nodes, Activation Function: {layer.activation.__class__.__name__}")
                print(f"  Weights Shape: {layer.W.shape}, Biases Shape: {layer.B.shape}")
                print(f"  Weights= : {layer.W}, Biases= {layer.B}")
            else:
                print(f"Layer {i}: Unknown Layer Type")

    def flatten_weights_and_biases(self):
        flattened_vector = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                # Flatten and append weights and biases of this layer
                flattened_weights = layer.W.flatten()
                flattened_biases = layer.B.flatten()
                flattened_vector.extend(flattened_weights.tolist())
                flattened_vector.extend(flattened_biases.tolist())
        return flattened_vector

### Reconstructing Weights and Biases from Flattened Vector

def unflatten_weights_and_biases(flattened_vector, configuration):
    position = []
    index = 0

    # Skip the input layer configuration, as it doesn't have weights or biases
    for nodes, _ in configuration[1:]:
        # Previous layer's node count is the number of rows for weights
        input_size = configuration[0] if not position else position[-1][0].shape[1]

        # Calculate the number of weights and biases
        num_weights = input_size * nodes
        num_biases = nodes

        # Extract weights and biases from the flattened vector
        weights = flattened_vector[index : index + num_weights]
        biases = flattened_vector[index + num_weights : index + num_weights + num_biases]

        # Reshape weights to the correct dimensions and add to position
        position.append((np.array(weights).reshape(input_size, nodes), np.array(biases)))

        # Update the index
        index += num_weights + num_biases

    return position

def fitness(X, Y, W):
    # Rebuild the neural network with the given weights
    position = unflatten_weights_and_biases(W, configuration)
    # print(position)
    neural_network = NeuralNetwork(configuration, position=position)

    # Perform forward pass
    predictions = neural_network.forward(X)

    # Calculate loss
    loss = loss_function.evaluate(predictions.flatten(), Y)
    return loss

### Function to Calculate Accuracy

def get_accuracy(Y, Y_pred):
    """
    Calcualtes accuracy.
    :param Y: int(N, )
        Correct labels.
    :param Y_pred: int(N, ) | double(N, C)
        Predicted labels of shape(N, ) or (N, C) in case of one-hot vector.
    :return: double
        Accuracy.
    """
    predicted_classes = (Y_pred >= 0.5).astype(int) # Convert probabilities to class labels
    return (Y == predicted_classes).mean()

class Particle:
    """
    Particle is a neural network representing a potential solution.
    """
    def __init__(self, num_dim, x_range, v_range, r1_range, r2_range, r3_range):
        """
        Particle class constructor
        :param num_dim: Number of dimensions.
        :param x_range: Range of dimension.
        :param v_range: Range of velocity.
        """

        self.x = np.random.uniform(x_range[0], x_range[1], (num_dim, )) # particle position
        self.v = np.random.uniform(v_range[0], v_range[1], (num_dim, )) # particle velocity
        self.pbest = np.inf                                             # personal best fitness
        self.pbestpos = np.zeros((num_dim, ))                           # personal best position
        self.informants_best_position = np.zeros((num_dim, ))           # informants best position
        self.informants = []                                            # particle's informants
        self.r1_range = r1_range                                        # array with min and max range for r1
        self.r2_range = r2_range                                        # array with min and max range for r2
        self.r3_range = r3_range                                        # array with min and max range for r3

    def update_informant_best(self, swarm):
        best_fitness = np.inf
        for informant in self.informants:
            if swarm[informant].pbest < best_fitness:
                best_fitness = swarm[informant].pbest
                self.informants_best_position = swarm[informant].pbestpos.copy()

    def update_velocity(self, global_best_position, alpha, beta, gamma, delta, r1_range, r2_range, r3_range):
        r1 = np.random.uniform(self.r1_range[0], self.r1_range[1])
        r2 = np.random.uniform(self.r2_range[0], self.r3_range[1])
        r3 = np.random.uniform(self.r3_range[0], self.r3_range[1])
        # r1, r2, r3 = np.random.rand(3)  # Random coefficients for stochastic components
        cognitive_component = beta * r1 * (self.pbestpos - self.x)
        social_component = gamma * r2 * (self.informants_best_position - self.x)
        global_component = delta * r3 * (global_best_position - self.x)
        self.v = alpha * self.v + cognitive_component + social_component + global_component


class Swarm:
    """
    The Swarm class is a collection of potential solutions, each represented by a particle.
    """
    def __init__(self, no_particle, num_dim, x_range, v_range, iw_range, c, num_informants, r1_range, r2_range, r3_range):
        """
        Swarm class constructor.
        :param no_particle:  Number of particles
        :param num_dim: Number of dimensions.
        :param x_range: Range of dimensions.
        :param v_range: Range of velocities.
        :param iw_range: Range of interia weights.
        :param c: c[0] -> cognitive parameter, c[1] -> social parameter, c[2] -> Global weight, c[3] -> Inertia weight
        :param num_informants: Number of informants
        """
        self.p = np.array([Particle(num_dim, x_range, v_range, r1_range, r2_range, r3_range) for i in range(no_particle)])
        self.gbest = np.inf
        self.gbestpos = np.zeros((num_dim, ))
        self.x_range = x_range
        self.v_range = v_range
        self.iw_range = iw_range
        self.c0 = c[0]            # Cognitive weight
        self.c1 = c[1]            # Social weight
        self.c2 = c[2]            # Global weight
        self.c3 = c[3]            # Inertia weight
        self.num_dim = num_dim    # Number of dimensions, in this case the total number of weights & biases
        self.num_informants = num_informants
        self.assign_informants()
        self.r1_range = r1_range
        self.r2_range = r2_range
        self.r3_range = r3_range

    def print_informants(self):
        """
        Function prints the informants of every particle.
        """
        for i, particle in enumerate(self.p):
            # Print the particle's index and its informants
            print(f"Particle {i} informants: {particle.informants}")

    def assign_informants(self):
        """
        Function assigns informants to every particle
        """
        for i, particle in enumerate(self.p):
            informants = set()
            # add informants until they are num_informants in informants set
            while len(informants) < self.num_informants:
                # Randomly select a potential informant
                possible_informant = np.random.randint(0, len(self.p))
                # if selected informant is not the particle itself, add it to set
                if possible_informant != i:
                    informants.add(possible_informant)
            particle.informants = np.array(list(informants))

    def optimize(self, function, X, Y,  print_step,  iter):
        """
        Function used to start optimization.
        :param function: function
            Function to be optimized
        :param X: input data
            Used in forward pass.
        :param Y: target class
            Used to calculate loss.
        :param print_step: int
            Step for printing
        :param iter: int
            Number of iterations.
        """
        for i in range(iter):
            for particle in self.p:
                # print("particle.x",particle.x)
                fitness = function(X, Y, particle.x)
                # print("fitness",fitness)
                # print("particle.pbest",particle.pbest)

                if fitness < particle.pbest:
                    particle.pbest = fitness
                    particle.pbestpos = particle.x.copy()

                if fitness < self.gbest:
                    self.gbest = fitness
                    self.gbestpos = particle.x.copy()

            for particle in self.p:
                # update informants best
                particle.update_informant_best(self.p)
                # update particle velocity
                particle.update_velocity(self.gbestpos, self.c3, self.c0, self.c1, self.c2, self.r1_range, self.r2_range, self.r3_range)
                # update particle position
                particle.x = particle.x +  particle.v

            if i % print_step == 0:
                print('iteration#: ', i+1,  ' loss: ', fitness)

        print("global best loss: ", self.gbest)

    def best_solution(self):
        '''
        return: array of parameters/weights.
        '''
        return self.gbestpos

def num_dim(configuration):
  total_weights = 0
  total_biases = 0

  # The number of nodes in the previous layer, initially the input layer
  prev_nodes = configuration[0]

  # Loop through each layer (excluding the input layer)
  for layer in configuration[1:]:
      # Extract the number of nodes in the current layer
      nodes = layer[0]

      # Calculate weights and biases for the current layer
      weights = prev_nodes * nodes
      biases = nodes

      # Add to total weights and biases
      total_weights += weights
      total_biases += biases

      # Update prev_nodes for the next iteration
      prev_nodes = nodes

  # Output the total number of dimensions
  total_dimensions = total_weights + total_biases
  return total_dimensions

### Running the ANN model with manual parameters

# def ann(config, X_train,  Y_train, no_solution,  w_range, lr_range, iw_range, c , num_informants, loss_function, iterations, r1_range, r2_range, r3_range):
def ann(params):
    # Unpack all parameters
    (
        config, X_train, Y_train, no_solution, w_range, lr_range,
        iw_range, c, num_informants, loss_function, iterations,
        r1_range, r2_range, r3_range
    ) = params

    # c[0] -> cognitive factor, c[1] -> global factor,  c[2] -> social factor, c[3] -> inertial weight
    no_dim = num_dim(config)

    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c, num_informants, r1_range, r2_range, r3_range)
    s.print_informants()

    # Drop column/s if user picks number of input nodes < number of features
    # Pick number of input nodes between 1 and num of features
    num_features = X_train.shape[1]  # number of features
    number_of_input_nodes = config[0]

    if (number_of_input_nodes < num_features):
      X_train = X_train.iloc[:, :number_of_input_nodes]
    # Train:Test Split
    test_size = 0.2  # User defined split ratio
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=test_size, random_state=42)

    print("\n")

    s.optimize(fitness, X_train, Y_train, no_solution, iterations)
    W = s.best_solution()

    # Perform forward pass with the best solution (weights)
    best_position = unflatten_weights_and_biases(W, config)
    best_nn = NeuralNetwork(config, position=best_position)

    Y_pred = best_nn.forward(X_train).flatten()
    Y_pred2 = best_nn.forward(X_test).flatten()

    # Calculate and print accuracy
    train_accuracy = get_accuracy(Y_train.flatten(), Y_pred)
    # # Calculate and print accuracy
    test_accuracy = get_accuracy(Y_test.flatten(), Y_pred2)

    return [train_accuracy, test_accuracy]   

params =  [layers, x_norm, y, 20, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.5, 0.3, 0.2, 0.9), 4, BinaryCrossEntropy(), 100, [0,1], [0,1], [0,1]]

def draw_neural_net(layers, left=.1, right=.9, bottom=.1, top=.9):
    '''
    Draw a neural network cartoon using matplotlib.

    :param layers: list of layers, each layer is a list of [number of nodes, activation function]
    :param left, right, bottom, top: coordinates to draw the diagram
    '''
    layer_sizes = [int(layer[0]) for layer in layers]  # Extracting number of nodes in each layer

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Define 15 pastel colors
    layer_colors = [
        '#FFD1DC', '#FFB347', '#FFFF99', '#CB99C9', '#C23B22',
        '#77DD77', '#AEC6CF', '#FDFD96', '#B39EB5', '#FFB7CE',
        '#FAE7B5', '#B0E0E6', '#FF6961', '#77DD77', '#CFCFC4'
    ]

    fig = plt.figure(figsize=(14, 13))
    ax = fig.gca()
    ax.axis('off')

    # Nodes and Labels
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color=layer_colors[n % 15], ec='k', zorder=4)
            ax.add_artist(circle)
        
        # Label Positioning
        if n == 0:  # Input Layer
            plt.text(n * h_spacing + left, top - 0.05, 'Input\nLayer', ha='center', va='top', fontsize=12)
        elif n == n_layers - 1:  # Output Layer
            plt.text(n * h_spacing + left, top - 0.05, 'Output\nLayer', ha='center', va='top', fontsize=12)
        else:  # Hidden Layers
            plt.text(n * h_spacing + left, top + 0.05, f'Hidden\nLayer {n}', ha='center', va='bottom', fontsize=12)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)


relu_activation = relu()
sigmoid_activation = Sigmoid()
tanh_activation = Tanh()


@app.route('/runAnn', methods=['POST'])
def runAnn():
    data = request.json
    print(data)
    no_solution = data['no_solution']
    # w_range = data['wRange']
    # lr_range = data['lrRange']
    iw_range = data['iw_range']
    c = data['c']
    num_informants = data['num_informants']
    iterations = data['num_iterations']
    r1_range = data['r1_Range']
    r2_range = data['r2_Range']
    r3_range = data['r3_Range']

    # Unpack all parameters
    (
        config, X_train, Y_train, no_solution, w_range, lr_range,
        iw_range, c, num_informants, loss_function, iterations,
        r1_range, r2_range, r3_range
    ) = params

    # c[0] -> cognitive factor, c[1] -> global factor,  c[2] -> social factor, c[3] -> inertial weight
    # send live print statements to the frontend
    acc2 = ann(params)
    return jsonify({
        "status": "success",
        "message": "ANN run",
        "trainAccuracy": acc2[0],
        "testAccuracy": acc2[1]
    })
    


@app.route('/createAnn', methods=['POST'])
def createAnn():
    data = request.json
    input_size = data['inputSize']
    hidden_layers = data['hiddenLayers']
    output_layer = data['outputLayer']
    loss_function = data['lossFunction']

    layers = [input_size, ]  # Input layer

    # Process hidden layers
    for layer in hidden_layers:
        neurons = layer['neurons']
        activation = get_activation_function(layer['activation'])
        layers.append([neurons, activation])

    # Output layer
    output_neurons = output_layer['neurons']
    output_activation = get_activation_function(output_layer['activation'])
    layers.append([output_neurons, output_activation])

    # Create and initialize your Neural Network here with 'layers'
    # nn = NeuralNetwork(layers)
    draw_neural_net(layers)
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    ann_details = {
        "input_size": input_size,
        "hidden_layers": hidden_layers,
        "output_layer": output_layer,
        "loss_function": loss_function
    }

    return jsonify({
        "status": "success", 
        "message": "ANN created", 
        "plot": plot_url,
        "ann_details": ann_details
    })

def get_activation_function(name):
    if name == 'relu':
        return relu()
    elif name == 'sigmoid':
        return Sigmoid()
    elif name == 'tanh':
        return Tanh()
    else:
        return Identity()  # Default or error handling

@app.route('/saveData', methods=['POST'])
def saveData(): 
    global x_norm, y # Declare global variables you want to modify

    file = request.files['file']
    normalized = request.form.get('normalize') == 'true'
    skipHeader = request.form.get('skipHeader') == 'true'
    testSplit = float(request.form.get('testSplit'))

    if skipHeader:
        df = pd.read_csv(file, skiprows=1)
    else:
        df = pd.read_csv(file, header=None)
    
    x_norm, y = seperateData(df)  # Assuming this function is correctly defined and splits the dataframe into features (X) and target (Y)
    
    if normalized:
        x_norm = normalize(x_norm)  # Normalize the features if required
    

    # Include additional response data
    return jsonify({
        "status": "success",
        "fileName": secure_filename(file.filename),
        "shape": df.shape,
        "xShape": x_norm.shape,
        "yShape": y.shape,
        "normalize": normalized,
        "splitRatio": testSplit
    })

if __name__ == '__main__':
    print("Starting server...")
    app.run(port=5000)
