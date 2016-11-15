package org.ml4bull.nn;

import org.ml4bull.ml.SupervisedAlgorithm;
import org.ml4bull.nn.layer.NeuronLayer;

public interface SupervisedNeuralNetwork extends SupervisedAlgorithm {
    MultiLayerPerceptron addHiddenLayer(NeuronLayer hiddenLayer);
}
