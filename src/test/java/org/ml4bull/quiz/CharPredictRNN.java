package org.ml4bull.quiz;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.ml4bull.algorithm.GradientDescent;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.data.WebBandog;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.RecurrentNeuronLayer;

import java.util.*;


@Slf4j
public class CharPredictRNN {

    @Test
    public void predictNextChars() {
        WebBandog webBandog = new WebBandog();
        String pageURL = "https://www.oxforddictionaries.com";
        Set<String> words = webBandog.findWords(pageURL, 30, 4);
        System.out.println(Objects.toString(words));
//        Set<String> words = new HashSet<>();
        String w1 = "hello";
        String w2 = "father";
        String w3 = "table";
        words.add(w1);
        words.add(w2);
        words.add(w3);



        MultiLayerPerceptron mlp = getNN();
//        mlp.addHiddenLayer(new HiddenNeuronLayer(50, new SigmoidFunction()));
//        mlp.addHiddenLayer(new HiddenNeuronLayer(26, new SigmoidFunction()));
//        mlp.addHiddenLayer(new HiddenNeuronLayer(7, new SigmoidFunction()));
        mlp.addHiddenLayer(new RecurrentNeuronLayer(7, null));

        double error;
        int epoch = 0;
        double maxError;
        double avgError;
        here:
        do {
            epoch++;
            error = 0;
            maxError = 0;
            String maxErrorWord = "";
            for (String word : words) {

                DataSet ds = createDataSetFromWord(word);
                double lastCharError = mlp.train(ds, false);
                if (Double.isNaN(lastCharError)) {
                    log.info("Word: {}", word);
                    log.error("NaN error...");
                    break here;
                }
                if (lastCharError > maxError) {
                    maxError = lastCharError;
                    maxErrorWord = word;
                }
                error += lastCharError;
            }
            avgError = error / words.size();

            log.info(String.format("Epoch %5d | Max Error: %.2f | word: %11s | AvgError: %.2f", epoch, maxError, maxErrorWord, avgError));
//            log.info("Max error {}", maxError);
        } while (avgError > 0.4);

        predict(mlp, w1);
        predict(mlp, w2);
        predict(mlp, w3);
    }

    @NotNull
    private double[][] predict(MultiLayerPerceptron mlp, String word) {
        DataSet wordDS = createDataSetFromWord(word);

        double[][] classify = mlp.classify(wordDS, false);

        for (double[] d : classify) {
            char c = fromDArrayToChar(d);
            log.info("Predicted char [{}]", c);
        }
        return classify;
    }

    private char fromDArrayToChar(double[] arr) {
        StepFunction sf = new StepFunction();
        double[] activate = sf.activate(arr);
        for (int i = 0; i < activate.length; i++) {
            if (activate[i] == 1) {
                return (char) (97 + i);
            }
        }
        if (arr[findMax(arr)] < 0.35) return '-';

        return (char) (findMax(arr) + 97);
    }

    private int findMax(double[] arr) {
        int maxValueIndex = 0;
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[maxValueIndex] < arr[i + 1]) {
                maxValueIndex = i + 1;
            }
        }
        return maxValueIndex;
    }

    @NotNull
    private DataSet createDataSetFromWord(String word) {
        List<double[]> list = wordToCharVector(word);
        double[][] input = new double[list.size()][];
        double[][] output = new double[list.size()][];
        for (int i = 0; i < list.size(); i++) {
            input[i] = list.get(i);
            output[i] = fromBinToDoubleArr(i != list.size() - 1 ? list.get(i + 1) : new double[25]);
        }

        return new DataSet(input, output);
    }

    private MultiLayerPerceptron getNN() {
        GradientDescent optAlg = GradientDescent.builder()
                .learningRate(1.8)
                .batchSize(20)
                .build();

        return MultiLayerPerceptron.builder()
                .outActFunc(new SoftmaxFunction())
                .input(7)
                .output(26)
                .optAlg(optAlg).build();
    }

    @Test
    public void tes() {
        System.out.println((int) 'a' - 97);
        System.out.println((int) 'z' - 97);
        System.out.println(Integer.toBinaryString('a'));
        System.out.println(Integer.toBinaryString('b'));
        System.out.println(Integer.toBinaryString('z'));
        System.out.println(Arrays.toString(convertCharToVector('c')));

        System.out.println(fromBinToInt(convertCharToVector('a')));
        System.out.println(fromBinToInt(convertCharToVector('z')));
        System.out.println(fromBinToInt(convertCharToVector('b')));

    }

    private double[] fromBinToDoubleArr(double[] bin) {
        double[] d = new double[26];
        int i = fromBinToInt(bin);

        if (i == -1) {
            return d;
        }

        d[i] = 1;
        return d;
    }

    private int fromBinToInt(double[] binary) {
        int k = 0;
        for (double v : binary) {
            k += v;
        }
        if (k == 0) {
            return -1;
        }
        int intRep = 0;
        for (int i = 0; i < binary.length - 1; i++) {
            intRep += Math.pow(binary[i] == 1 ? 2 : 0, binary.length - i - 1);
        }
        intRep += binary[binary.length - 1];

        return intRep - 97;
    }

    private List<double[]> wordToCharVector(String word) {
        List<double[]> wordVector = new ArrayList<>(word.length());
        for (char c : word.toCharArray()) {
            wordVector.add(convertCharToVector(c));
        }
        return wordVector;
    }

    private double[] convertCharToVector(char c) {
        double[] charArr = new double[7];
        String binary = Integer.toBinaryString(c);
        char[] chars = binary.toCharArray();
//        System.out.println(Arrays.toString(chars) + " " + c);
        for (int i = 0; i < chars.length; i++) {
            charArr[i] = chars[i] == '1' ? 1 : 0;
        }
        return charArr;
    }
}
