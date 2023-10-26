# Function to create and compile ANN using information obtained from INPUT file

def f_ANN_model(input_file_data): 
    
    """
    Creates ANN model using input details extracted from "Input_ANN.txt" file
    """

    import tensorflow as tf;
    from tensorflow.keras import layers;
    from tensorflow.keras import Model;
    from tensorflow.keras import optimizers;


    #extract details from "Input_ANN.txt" file
    n_feats = len(input_file_data['x']); #no. of features
    Optimizer = input_file_data['optimizer'][0];
    lr = float(input_file_data['learning_rate'][0]);
    Loss = input_file_data['loss_function'][0];
    layer_units = input_file_data['layer_units'];
    layer_activation = input_file_data['activation_functions'];


    feature_input = layers.Input(shape = n_feats); #ANN input layer
    
    x = feature_input;
    
    # Hidden Layers
    for h in range(0, len(layer_units)-1):
        x = layers.Dense(layer_units[h], activation=layer_activation[h])(x);

    # Create output layer with 1 unit and activation function mentioned in INPUT file
    output_layer = layers.Dense(layer_units[-1], activation=layer_activation[-1])(x);

    
    # Create model:  # input = features  # output = binary vector
    model = Model(feature_input, output_layer);

    # Display model summary
    model.summary()
    
    # Compiling the model using loss and optimizer given in INPUT file
    model.compile(loss=str(Loss), optimizer=getattr(optimizers, Optimizer)(learning_rate=lr), metrics=['mae']);
    
    return model