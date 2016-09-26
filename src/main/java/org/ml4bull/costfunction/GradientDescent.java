package org.ml4bull.costfunction;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.List;

public class GradientDescent {

    private double learningRate;
    private double regularization;
    private ActivationFunction af;

    public GradientDescent(double learningRate, double regularization, ActivationFunction af) {
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.af = af;
    }

    public void calculate(List<Neuron> neurons, double[][][] partialD, double learningRate) {
        neurons.forEach(n -> {
            double[] weights = n.getWeights();

        });

    }

    public void calculate(double[][] data, double[][] expected, double[] theta) {
        MatrixOperations mo = Factory.getMatrixOperations();
        int m = data.length;

        for (int j = 0; j < theta.length; j++) {
            double cost = 0;
            for (int i = 0; i < m; i++) {
                double qtx = mo.multiply(theta, data[i]);
                double h0 = af.activate(qtx);
                // simply one Y
                cost += (h0 - expected[i][0]) * data[i][j];
            }
            if (j != 0) {
                theta[j] *= (1 - learningRate * regularization / m);
            }

            theta[j] -= learningRate * cost / m;
        }
    }
}
