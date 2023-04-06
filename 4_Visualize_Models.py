from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from config.Example_Config_Model_Visualizer import get_model_viz_config
from constants.AI_params import *
from models.modelSelector import select_2d_model
from models.model_viz import print_layer_names, plot_cnn_filters_by_layer, plot_intermediate_2dcnn_feature_map
import matplotlib.image as mpltimage
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    config = get_model_viz_config()

    model_weights_file = config[ClassificationParams.model_weights_file]
    model = select_2d_model(config)
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # # Gets all the config
    model_config = model.get_config()

    # # All Number of parameters
    print(F' Number of parameters: {model.count_params()}')
    # Number of parameters by layer
    # print(F' Number of parameters first CNN: {model.layers[3].count_params()}')

    # Example of plotting the filters of a single layer
    print_layer_names(model)
    plot_cnn_filters_by_layer(model.layers[1], 'Sobel filter learned :) ')  # The harcoded 1 should change by project

    # ========= Here you need to build your test input different in each project ====
    input_file = '/home/olmozavala/Dropbox/MyProjects/OZ_LIB/AI_Template/TESTDATA/033.jpg'
    img = mpltimage.imread(input_file)
    input_array = np.expand_dims(np.expand_dims(img[:, :, 0], axis=2), axis=0)
    # ========= Here you need to build your test input different in each project ====

    # If you want to show the NN prediction
    output_NN = model.predict(input_array, verbose=1)
    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(output_NN[0,:,:,0])
    plt.show()

    # =========== Output from the last layer (should be the same as output_NN
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers[1:]]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([input_array]) for func in functors]

    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(layer_outs[0][0][0,:,:,0])
    plt.show()


    # ============Output from intermediate layers
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('conv2d').output)

    intermediate_output = intermediate_layer_model.predict(input_array)
    plot_intermediate_2dcnn_feature_map(intermediate_output,  title='Intermediate layer')
