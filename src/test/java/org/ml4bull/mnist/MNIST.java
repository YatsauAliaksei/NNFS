package org.ml4bull.mnist;

import com.google.common.base.Stopwatch;
import lombok.extern.log4j.Log4j2;
import org.junit.Test;
import org.ml4bull.algorithm.ReLUFunction;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.algorithm.optalg.RMSPropGradientDescent;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.HiddenNeuronLayer;
import org.ml4bull.util.MLUtils;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

@Log4j2
public class MNIST {

    private int reduceDSScale = 100;

    @Test
    public void main() throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        DataSet trainDS = getTrainDS();
        System.out.println("Watch: " + stopwatch.toString());
        log.info("Train data set loaded.");

        double[] sample = trainDS.getInput()[400];
        double[] sampleLabel = trainDS.getOutput()[400];

        printSample(sample, sampleLabel);

        GradientDescent optAlg = RMSPropGradientDescent.build()
                .learningRate(0.01)
                .withRegularization(true)
                .batchSize(50)
                .build();

        MultiLayerPerceptron sp = MultiLayerPerceptron.builder()
                .input(28 * 28)
                .output(10)
                .outActFunc(new SoftmaxFunction())
                .optAlg(optAlg)
                .build();

        sp.addHiddenLayer(new HiddenNeuronLayer(300, new SigmoidFunction()));
//        sp.addHiddenLayer(new HiddenNeuronLayer(30, new HyperbolicTangentFunction()));
//        sp.addHiddenLayer(new HiddenNeuronLayer(300, new ReLUFunction()));
        normalize(trainDS);

        double error;
        int epoch = 0;
        do {
            error = sp.train(trainDS, true);
            log.info("Epoch: {} | Error: {}", ++epoch, +error);
        } while (error > 1e-2);

        DataSet testDS = getTestDS();
        log.info("Test data set loaded.");

        normalize(testDS);

        StepFunction stepFunction = new StepFunction();
        AtomicInteger success = new AtomicInteger();
        AtomicInteger total = new AtomicInteger();
        sp.classify(testDS, false,
                (i, calc, ideal) -> {
                    int predicted = MLUtils.transformClassToInt(stepFunction.activate(calc));
                    int real = MLUtils.transformClassToInt(ideal);
                    System.out.println(i + ". " + "Calculated:" + predicted + ". Real: " + real);
                    if (real == predicted) {
                        success.addAndGet(1);
                    }
                    total.addAndGet(1);
                });

        log.info("True Positive: {}%", String.format("%.2f", success.get() / (double) total.get() * 100));
        log.info("Error rate: {}%", String.format("%.2f", (total.get() - success.get()) / (double) total.get()));
        log.info("Time overall {}", stopwatch.stop());

        System.exit(0);
    }

    public static void printSample(double[] sample, double[] sampleLabel) {
        System.out.println("Label: " + (MLUtils.transformClassToInt(sampleLabel) - 1));

        for (int i = 0; i < sample.length; i++) {
            if (i % 28 == 0)
                System.out.println();

            System.out.print((int) sample[i] + " ");
        }
        System.out.println();
    }

    private void normalize(DataSet trainDS) {
        for (double[] image : trainDS.getInput()) {
            for (int i = 0; i < image.length; i++) {
                image[i] /= 255;
            }
        }
    }

    private DataSet getTrainDS() {
        double[][] images = readImages("train-images.idx3-ubyte");
        double[][] labels = readLabels("train-labels.idx1-ubyte");
        return new DataSet(images, labels);
    }

    private DataSet getTestDS() {
        double[][] images = readImages("t10k-images.idx3-ubyte");
        double[][] labels = readLabels("t10k-labels.idx1-ubyte");
        return new DataSet(images, labels);
    }

    private double[][] readLabels(String filename) {
        return DataSetLoader.load("/home/mrj/Downloads/MNIST/" + filename, magic -> magic == 2049,
                dis -> {
                    try {
                        int numberOfLabels = dis.readInt() / reduceDSScale;

                        double[][] result = new double[numberOfLabels][];
                        for (int i = 0; i < numberOfLabels; i++) {
                            int labelNumber = dis.readUnsignedByte();
//                            System.out.println(labelNumber);
                            result[i] = MLUtils.transformIntToClass(labelNumber + 1, 10);
                        }

                        return result;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
    }

    private double[][] readImages(String filename) {
        return DataSetLoader.load("/home/mrj/Downloads/MNIST/" + filename, magic -> magic == 2051,
                dis -> {
                    try {
                        int numberOfImages = dis.readInt() / reduceDSScale;
                        int numberOfRows = dis.readInt();
                        int numberOfColumns = dis.readInt();

                        double[][] result = new double[numberOfImages][];
                        int imagePixelSize = numberOfColumns * numberOfRows;
                        for (int i = 0; i < numberOfImages; i++) {
                            double[] image = new double[imagePixelSize];
                            for (int pointer = 0; pointer < imagePixelSize; pointer++) {
                                double pixel = dis.readUnsignedByte();
                                image[pointer] = pixel;
                            }
                            result[i] = image;
                        }

                        return result;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
    }
}
