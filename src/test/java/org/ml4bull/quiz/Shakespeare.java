package org.ml4bull.quiz;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.junit.Test;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.algorithm.optalg.RMSPropGradientDescent;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.RecurrentNeuronLayer;
import org.ml4bull.util.MLUtils;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static org.assertj.core.api.Assertions.assertThat;

@Log4j2
public class Shakespeare {

    @Test
    public void charsTest() {
        System.out.println("Space: " + (int) ' ');

        System.out.println("{: " + '{');
        for (int i = 'a'; i <= 'z'; i++)
            System.out.println(((char) i) + " - " + (i - 96));

        System.out.println("AxA12 -$ /{1fas!fR}".replaceAll("[^a-z ]", ""));

        double[] doubles = charToVector('a');
        char a = vectorToChar(doubles);

        assertThat(a).isEqualTo('a');

        doubles = charToVector(' ');
        char space = vectorToChar(doubles);

        assertThat(space).isEqualTo(' ');
    }

    @Test
    public void main() {
        GradientDescent optAlg = RMSPropGradientDescent.build()
                .learningRate(0.1)
                .batchSize(20)
                .build();

        MultiLayerPerceptron mlp = MultiLayerPerceptron.builder()
                .outActFunc(new SoftmaxFunction())
                .input(27)
                .output(27)
                .optAlg(optAlg).build();

        mlp.addHiddenLayer(new RecurrentNeuronLayer(new HyperbolicTangentFunction(), 100, 20));

        DataSet trainDS = createDS();
        log.info("Train data set created. Size: {}", trainDS.getInput().length);
        double error;
        int epoch = 0;
        int k = 0;
        do {
            error = mlp.train(trainDS, false);
            log.info("Epoch: {} | Error: {}", ++epoch, +error);

            if (k++ % 5 == 0) {  // each 5 times
                testing(mlp);
            }
        } while (error > 1e-1);

        log.info("Starting classification...");

        testing(mlp);
    }

    private void testing(MultiLayerPerceptron mlp) {
        int startLetter = ThreadLocalRandom.current().nextInt('a', 'z');

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

    private DataSet createDS() {
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

    private char vectorToChar(double[] v) {
        int maxIndex = 0;
        for (int i = 0; i < v.length; i++) {
            if (v[i] > v[maxIndex])
                maxIndex = i;
        }

        double[] predictedV = new double[27];
        predictedV[maxIndex] = 1;

        if (maxIndex == 0)
            return ' ';

        return (char) (MLUtils.transformClassToInt(predictedV) + 96);
    }

    private double[][] wordToVector(String word) {
        char[] chars = word.toCharArray();
        double[][] result = new double[chars.length + 1][];
        for (int i = 0; i < chars.length; i++) {
            result[i] = charToVector(chars[i]);
        }
        result[result.length - 1] = MLUtils.transformIntToClass(27, 27); // space
        return result;
    }

    private double[] charToVector(char c) {
        if (c == ' ') {
            return MLUtils.transformIntToClass(27, 27); // space
        }
        return MLUtils.transformIntToClass(c - 96, 27);
    }

    // a-z
    @SneakyThrows
    private char[] shakespeareArt() {
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
//        String url = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt";

        String tempDir = System.getProperty("java.io.tmpdir");
//        String fileLocation = tempDir + "/Shakespeare.txt";    //Storage location from downloaded file
        String fileLocation = tempDir + "/shakespeare.txt";    //Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            log.info("File downloaded to " + f.getAbsolutePath());
        } else {
            log.info("Using existing text file at " + f.getAbsolutePath());
        }
        List<String> lines = Files.readAllLines(f.toPath());

        return lines.stream()
                .filter(StringUtils::isNotBlank)
                .filter(l -> ThreadLocalRandom.current().nextInt(0, 10) > 8)
//                .flatMap(line -> Splitter.on(" ").trimResults().splitToList(line).stream())
                .map(String::toLowerCase)
                .map(word -> word.replaceAll("[^a-z ]", ""))
//                .filter(StringUtils::isNotBlank)
                .reduce((l1, l2) -> l1 + " " + l2)
                .orElseThrow(RuntimeException::new)
                .toCharArray();
    }
}
