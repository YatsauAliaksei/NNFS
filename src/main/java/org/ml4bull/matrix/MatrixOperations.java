package org.ml4bull.matrix;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public interface MatrixOperations {
    Logger log = LoggerFactory.getLogger(MatrixOperations.class);

    double[][] transpose(double[][] matrix);

    double[][] transpose(double[] matrix);

    void scalarMultiply(double[][] matrix, double k);

    default void printMatrix(double[][] matrix) {
        Arrays.stream(matrix).forEachOrdered(m -> {
            String line = Arrays.stream(m)
                    .mapToObj(String::valueOf)
                    .collect(Collectors.joining(" "));
            log.info(line);
        });
    }

    default void printMatrix(double[] matrix) {
        DoubleStream.of(matrix).boxed().forEachOrdered(m -> log.info(m.toString()));
    }

    void scalarMultiply(double[] matrix, double k);

    /**
     * Matrix multiplication. {@param matrix1, matrix2} should be same length.
     * @param matrix1
     * @param matrix2
     * @return
     */
    double multiply(double[] matrix1, double[] matrix2);

    double[] scalarMultiply(double[] matrix1, double[] matrix2);

    double[][] multiply(double[][] matrix1, double[][] matrix2);

    double[] multiplySingleDim(double[][] matrix1, double[] matrix2);

    void roundMatrix(double[] matrix, double threshold);

    void addition(double[][] matrix, double value);

    double[][] getFullIdentityMatrix(int column, int row);

    double[][] sum(double[][] delta, double[][] tmpE);

    double[] sum(double[] el1, double[] el2);

    double[] copy(double[] source);

    boolean same(double[] i1, double[] i2);

    double[] concatenate(double[] m1, double[] m2);
}
