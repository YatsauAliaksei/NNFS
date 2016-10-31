package org.ml4bull.quiz;

import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.matrix.DoubleIterator;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;
import org.ml4bull.nn.layer.HiddenNeuronLayer;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static java.lang.System.nanoTime;

public class MazeNN {


    public static void main(String[] arg) {
        MazeNN mazeNN = new MazeNN();
        MultiLayerPerceptron sp = new MultiLayerPerceptron(100, 1, new SigmoidFunction());
        sp.addHiddenLayer(new HiddenNeuronLayer(10, new SigmoidFunction()));
        sp.addHiddenLayer(new HiddenNeuronLayer(10, new SigmoidFunction()));
        sp.setLearningRate(5e-1).setRegularizationRate(11e-1);

        DataSet trainSet = mazeNN.getTrainSet();
        double error;
        int epoch = 0;
        do {
            long start = nanoTime();
            error = sp.train(trainSet);
            System.out.println("Epoch: " + ++epoch + " | Error: " + error + " - " + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));
        } while (error > 5e-1);

        DataSet testSet = mazeNN.getTestSet();
        sp.test(testSet, mazeNN.getResultProcessor());
        System.exit(0);
    }

    public DataSet getTestSet() {
        return getDataSet(100);
    }

    public DataSet getTrainSet() {
        return getDataSet(5_000);
    }

    private DataSet getDataSet(int size) {
        double[][] input = new double[size][];
        double[][] output = new double[size][];

        int negative = 0;
        int counter = 0;
        while (size != counter) {
            String[][] maze = MazeUtils.generateMaze(10, 10);
            boolean unreachable = MazeUtils.isUnreachable(maze);

            if (unreachable && negative++ >= size / 2) continue;

            input[counter] = transformMaze(maze);
            output[counter++] = new double[]{unreachable ? 0.0 : 1.0};
        }

        return new DataSet(input, output);
    }

    private double[] transformMaze(String[][] maze) {
        final double[] in = new double[maze.length * maze[0].length];

        ((DoubleIterator) (l, e) -> {
            if (maze[l][e].equals("X")) {
                in[l * maze.length + e] = 1;
            } else
                in[l * maze.length + e] = 0;

        }).iterate(maze);

        return in;
    }

    public Printer getResultProcessor() {
        return (iteration, predict, ideal) -> {
            double[] roundedAnswer = new StepFunction().activate(predict);
            if (!Arrays.equals(ideal, roundedAnswer)) {
                System.out.println(iteration + " Error");
            }
        };
    }
}