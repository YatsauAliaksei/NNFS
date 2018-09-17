package org.ml4bull.quiz;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.algorithm.optalg.RMSPropGradientDescent;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.LSTMNeuronLayer;
import org.ml4bull.nn.layer.LSTMNeuronParallelLayer;
import org.ml4bull.nn.layer.LinearNeuronLayer;
import org.ml4bull.util.CharUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

@Log4j2
public class ShakespeareLSTM {

    // russian
    private static CharUtils.Language language = CharUtils.Language.RUSSIAN;


    public static void main(String[] args) {
        GradientDescent optAlg = RMSPropGradientDescent.buildRMS()
//        GradientDescent optAlg = GradientDescent.builder()
//        GradientDescent optAlg = ADAMGradientDescent.buildAdam()
                .learningRate(0.01)
                .withRegularization(false)
                .batchSize(10)
                .build();

        MultiLayerPerceptron mlp = MultiLayerPerceptron.builder()
                .outputLayer(new LinearNeuronLayer())
                .input(language.getCharNum())
                .output(language.getCharNum())
                .optAlg(optAlg).build();

        LSTMNeuronLayer lstmNeuronLayer = new LSTMNeuronLayer(100, language.getCharNum(), null, 10, 10);
        mlp.addHiddenLayer(lstmNeuronLayer);

        DataSet trainDS = createDS();
        log.info("=============================================\n" +
                "Train data set created. Size: {}", trainDS.getInput().length);

        mlp.training(trainDS, 1e-1, (e) -> testing(mlp));

        log.info("Starting testing...");

        testing(mlp);
    }

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
        return CharUtils.vectorToChar(v, language);
    }

    private static double[] charToVector(char c) {
        return CharUtils.charToVector(c, language);
    }

    // a-z
    @SneakyThrows
    private static char[] shakespeareArt() {
//        String s = Japan.japan.toLowerCase().replaceAll("[^а-я ]", "");
//        log.info(s);
//        return s.toCharArray();

        List<String> lines = readFile();
        return lines.stream()
                .filter(StringUtils::isNotBlank)
//                .map(word -> word.toLowerCase().replaceAll("[^a-z ]", ""))
                .map(word -> word.toLowerCase().replaceAll("[^а-я ]", "")) // russian
                .map(String::trim)
                .peek(log::info)
                .reduce((l1, l2) -> l1 + " " + l2)
                .orElseThrow(RuntimeException::new)
                .toCharArray();
    }

    private static List<String> readFile() throws IOException {
                String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
//        String url = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt";
//        String url = "https://sherlock-holm.es/stories/plain-text/cano.txt";

//        String tempDir = System.getProperty("java.io.tmpdir");
        String tempDir = System.getProperty("user.home");
//        String fileLocation = tempDir + "/shakespeare.txt";    //Storage location from downloaded file
//        String fileLocation = tempDir + "/sherlock.txt";    //Storage location from downloaded file
        String fileLocation = tempDir + "/chebur.txt";    //Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            log.info("File downloaded to " + f.getAbsolutePath());
        } else {
            log.info("Using existing text file at " + f.getAbsolutePath());
        }
        return Files.readAllLines(f.toPath());
    }
}
