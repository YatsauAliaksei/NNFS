package org.ml4bull.quiz;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.algorithm.optalg.RMSPropGradientDescent;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.LinearNeuronLayer;
import org.ml4bull.nn.layer.RecurrentNeuronLayer;
import org.ml4bull.util.MLUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;

@Log4j2
public class ShakespeareRNN {

    // russian
    private static int charNum = 33; // plus space
    private static int leading = 1071;

    // english
//    private static int charNum = 27;
//    private static int leading = 96;

    public static void main(String[] args) {
        GradientDescent optAlg = RMSPropGradientDescent.build()
                .learningRate(0.01)
                .batchSize(10)
                .build();

        MultiLayerPerceptron mlp = MultiLayerPerceptron.builder()
                .outputLayer(new LinearNeuronLayer())
                .input(charNum)
                .output(charNum)
                .optAlg(optAlg).build();

        mlp.addHiddenLayer(new RecurrentNeuronLayer(100, charNum, 10));

        DataSet trainDS = createDS();
        log.info("=============================================\n" +
                "Train data set created. Size: {}", trainDS.getInput().length);

        mlp.trainAsync(trainDS, 1e-1, (e) -> testing(mlp));

/*        int processors = Runtime.getRuntime().availableProcessors();
        log.info("Processors count [{}]", processors);

        CompletableFuture[] cfs = IntStream.range(0, processors)
                .boxed()
                .map(i -> CompletableFuture.runAsync(() -> training(mlp, trainDS))).toArray(CompletableFuture[]::new);

        CompletableFuture.allOf(cfs).join();*/

        log.info("Starting testing...");

        testing(mlp);
    }

/*    private static void training(MultiLayerPerceptron mlp, DataSet trainDS) {
        double error;
        int epoch = 0;
        int k = 0;
        do {
            error = mlp.train(trainDS, false);
            log.info("Epoch: {} | Error: {}", ++epoch, +error);

            if (k++ % 2 == 0) {  // each k times
                testing(mlp);
            }
        } while (error > 1e-1);
    }*/

    private static void testing(MultiLayerPerceptron mlp) {
        int startLetter = ThreadLocalRandom.current().nextInt('а', 'я' + 1); // russian
//        int startLetter = ThreadLocalRandom.current().nextInt('a', 'z' + 1);

        double[] v = charToVector((char) startLetter); // starting letter

        System.out.println("========================== TESTING ===========================");
        int k = 500;
        while (k-- > 0) {
            System.out.print(vectorToChar(v));
            v = mlp.classify(v);
        }
        System.out.println();
        System.out.println("==============================================================");
    }

    private static DataSet createDS() {
        char[] text = shakespeareArt();

        double[][] input = new double[text.length][];
        double[][] output = new double[text.length][];
        for (int i = 0; i < text.length - 1; i++) {
            input[i] = charToVector(text[i]);
            output[i] = charToVector(text[i + 1]);
        }
        input[text.length - 1] = charToVector(text[text.length - 1]);
        output[text.length - 1] = charToVector(' ');

        return new DataSet(input, output);
    }

    private static char vectorToChar(double[] v) {
        int maxIndex = 0;
        for (int i = 0; i < v.length; i++) {
            if (v[i] > v[maxIndex])
                maxIndex = i;
        }

        double[] predictedV = new double[charNum];
        predictedV[maxIndex] = 1;

        if (maxIndex == 0)
            return ' ';

        return (char) (MLUtils.transformClassToInt(predictedV) + leading);
    }

    private static double[] charToVector(char c) {
        return MLUtils.transformIntToClass(c == ' ' ? charNum : c - leading, charNum);
    }

    // a-z
    @SneakyThrows
    private static char[] shakespeareArt() {
        List<String> lines = readFile();
        return lines.stream()
//                .map(lines::get)
//                .map(word -> word.toLowerCase().replaceAll("[^a-z ]", ""))
                .map(word -> word.toLowerCase().replaceAll("[^а-я ]", "")) // russian
//                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
//                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
//                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
                .map(String::trim)
                .peek(log::info)
                .reduce((l1, l2) -> l1 + " " + l2)
                .orElseThrow(RuntimeException::new)
                .toCharArray();

/*        return lines.stream()
                .filter(StringUtils::isNotBlank)
                .filter(l -> l.length() < 70)
                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
//                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
//                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
//                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 4)
//                .flatMap(line -> Splitter.on(" ").trimResults().splitToList(line).stream())
                .map(String::toLowerCase)
                .map(word -> word.replaceAll("[^a-z ]", ""))
                .map(String::trim)
                .peek(log::info)
//                .filter(StringUtils::isNotBlank)
                .reduce((l1, l2) -> l1 + " " + l2)
                .orElseThrow(RuntimeException::new)
                .toCharArray();*/
    }

    private static List<String> readFile() throws IOException {
//                String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
//        String url = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt";
//        String url = "https://sherlock-holm.es/stories/plain-text/cano.txt";

//        String tempDir = System.getProperty("java.io.tmpdir");
        String tempDir = System.getProperty("user.home");
//        String fileLocation = tempDir + "/shakespeare.txt";    //Storage location from downloaded file
//        String fileLocation = tempDir + "/sherlock.txt";    //Storage location from downloaded file
        String fileLocation = tempDir + "/chebur.txt";    //Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
//            FileUtils.copyURLToFile(new URL(url), f);
            log.info("File downloaded to " + f.getAbsolutePath());
        } else {
            log.info("Using existing text file at " + f.getAbsolutePath());
        }
        return Files.readAllLines(f.toPath());
    }

    @Test
    public void russianChars() {
        System.out.println((int) 'а' + "-" + (int) 'я');
        AtomicInteger k = new AtomicInteger();
        IntStream.rangeClosed(1072, 1103)
                .peek(i -> k.incrementAndGet())
                .forEach(i -> System.out.println((char) i + " - " + (i - leading)));

        System.out.println("Total:" + k);

        double[] doubles = charToVector('а'); // russian A
        char a = vectorToChar(doubles);

        assertThat(a).isEqualTo('а');

        doubles = charToVector(' ');
        char space = vectorToChar(doubles);

        assertThat(space).isEqualTo(' ');

    }

    @Test
    public void charsTest() {
        System.out.println("Space: " + (int) ' ');

        for (int i = 'a'; i <= 'z'; i++)
            System.out.println(((char) i) + " - " + (i - 96));

        System.out.println("AxA12 -$ /{1fas!fR}".replaceAll("[^а-я ]", ""));

        double[] doubles = charToVector('a');
        char a = vectorToChar(doubles);

        assertThat(a).isEqualTo('a');

        doubles = charToVector(' ');
        char space = vectorToChar(doubles);

        assertThat(space).isEqualTo(' ');
    }

    private double[][] wordToVector(String word) {
        char[] chars = word.toCharArray();
        double[][] result = new double[chars.length + 1][];
        for (int i = 0; i < chars.length; i++) {
            result[i] = charToVector(chars[i]);
        }
        result[result.length - 1] = MLUtils.transformIntToClass(charNum, charNum); // space
        return result;
    }
}
