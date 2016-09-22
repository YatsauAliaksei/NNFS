package org.ml4bull.matrix;

import java.util.stream.DoubleStream;

public interface MatrixOperations {

    double[][] transpose(double[][] matrix);

    double[][] transpose(double[] matrix);

    void multiply(double[][] matrix, double k);

    default void printMatrix(double[][] matrix) {
        for (double[] m : matrix) {
            DoubleStream.of(m).forEach(e -> System.out.print(e + " "));
            System.out.println();
        }
    }

    default void printMatrix(double[] matrix) {
        DoubleStream.of(matrix).forEach(System.out::println);
    }

    double[][] multiply(double[][] matrix1, double[][] matrix2);
}
