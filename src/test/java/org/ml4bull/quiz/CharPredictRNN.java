package org.ml4bull.quiz;

import com.google.common.base.Splitter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.algorithm.optalg.RMSPropGradientDescent;
import org.ml4bull.bot.TelegramBot;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.RecurrentNeuronLayer;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;


@Slf4j
public class CharPredictRNN {

    @Test
    public void predictNextChars() throws IOException {
//        Set<String> words = shakespeareDS();
        log.info("Data set has been created.");

//        WebBandog webBandog = new WebBandog();
//        String pageURL = "https://www.oxforddictionaries.com";
//        Set<String> words = webBandog.findWords(pageURL, 30, 5);
//        System.out.println(Objects.toString(words));
        Set<String> words = new HashSet<>();
//        int[] arrs = new int[] {1, 2, 3};
//        List<int[]> coins = Arrays.asList(arrs);
        String w1 = "hello";
//        String w1 = "hellofathetable";
//        String w2 = "abcdefghiklmnop";
//        String w3 = "howareyoumanoop";

        words.add(w1);
//        words.add(w2);
//        words.add(w3);

        MultiLayerPerceptron mlp = getNN();
//        mlp.addHiddenLayer(new HiddenNeuronLayer(7, new HyperbolicTangentFunction()));
//        mlp.addHiddenLayer(new HiddenNeuronLayer(26, new SigmoidFunction(), false));
//        mlp.addHiddenLayer(new HiddenNeuronLayer(26, new LiniarFunction(), false));
//        mlp.addHiddenLayer(new RecurrentNeuronLayer(new HyperbolicTangentFunction(), 2));
        mlp.addHiddenLayer(new RecurrentNeuronLayer(new SigmoidFunction(), 2));
//        mlp.addHiddenLayer(new LSTMNeuronLayer(new SigmoidFunction(), 2));
//        mlp.addHiddenLayer(new LSTMNeuronLayer(new HyperbolicTangentFunction(), 2));
//        mlp.addHiddenLayer(new HiddenNeuronLayer(26, new SigmoidFunction(), false));

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
                if (ds.getDataSetSize() == 0) {
                    log.info("Omitting word [{}]", word);
                    continue;
                }

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
        } while (avgError > 0.4);

        predict(mlp, "he");
        predict(mlp, "hel");
        predict(mlp, "hell");
        predict(mlp, "ye");
    }

    private Set<String> shakespeareDS() throws IOException {
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/Shakespeare.txt";    //Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }
        List<String> lines = Files.readAllLines(f.toPath());
        return lines.stream()
                .flatMap(line -> Splitter.on(" ").trimResults().splitToList(line).stream())
                .collect(Collectors.toSet());
    }

    private MultiLayerPerceptron getNN() {
        GradientDescent optAlg = RMSPropGradientDescent.builder()
                .learningRate(0.1)
                .batchSize(15)
                .build();

        return MultiLayerPerceptron.builder()
                .outActFunc(new SoftmaxFunction())
                .input(26)
                .output(26)
                .optAlg(optAlg).build();
    }

    @NotNull
    private double[][] predict(MultiLayerPerceptron mlp, String word) {
        DataSet wordDS = createDataSetFromWord(word);

        double[][] classify = mlp.classify(wordDS, false);

        for (double[] d : classify) {
            char c = fromDArrayToChar(d);
            log.info("Predicted char [{}]", c);
            TelegramBot bot = TelegramBot.takeMe("firstOne123bot");
            bot.say("Predicted char [" + c + "]");
        }
        log.info("=====================================");
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
//            input[i] = list.get(i);
            input[i] = fromBinToDoubleArr(list.get(i));
            output[i] = fromBinToDoubleArr(i != list.size() - 1 ? list.get(i + 1) : new double[25]);
        }

        return new DataSet(input, output);
    }

    @Test
    public void tes() {
        List<double[]> abc = wordToCharVector("abs");
        for (double[] c : abc) {
            System.out.println(Arrays.toString(c));
        }
        System.out.println((int) 'a' - 97);
        System.out.println((int) 'z' - 97);
        System.out.println(Integer.toBinaryString('a'));
        System.out.println(Integer.toBinaryString('b'));
        System.out.println(Integer.toBinaryString('z'));
        System.out.println(Arrays.toString(convertCharToVector('c')));

        System.out.println(fromBinToInt(convertCharToVector('a')));
        System.out.println(fromBinToInt(convertCharToVector('a')));
        System.out.println(fromBinToInt(convertCharToVector('z')));
        System.out.println(fromBinToInt(convertCharToVector('b')));

    }

    private double[] fromBinToDoubleArr(double[] bin) {
        double[] d = new double[26];
        int i = fromBinToInt(bin);

        if (i < 0) {
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
        word = word.replaceAll("[^a-zA-Z]", "");
        for (char c : word.toLowerCase().toCharArray()) {
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
