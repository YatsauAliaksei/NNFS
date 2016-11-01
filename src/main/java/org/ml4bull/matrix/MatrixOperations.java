package org.ml4bull.matrix;

import com.sun.istack.internal.NotNull;

import java.util.stream.DoubleStream;

public interface MatrixOperations {

    double[][] transpose(double[][] matrix);

    double[][] transpose(double[] matrix);

    void scalarMultiply(double[][] matrix, double k);

    default void printMatrix(double[][] matrix) {
        for (double[] m : matrix) {
            DoubleStream.of(m).forEach(e -> System.out.print(e + " "));
            System.out.println();
        }
    }

    default void printMatrix(double[] matrix) {
        DoubleStream.of(matrix).forEach(System.out::println);
    }

    /**
     * Matrix multiplication. {@param matrix1, matrix2} should be same length.
     * @param matrix1
     * @param matrix2
     * @return
     */
    double multiply(@NotNull double[] matrix1, @NotNull double[] matrix2);

    double[][] multiply(double[][] matrix1, double[][] matrix2);

    double[] multiplySingleDim(double[][] matrix1, double[] matrix2);

    void roundMatrix(double[] matrix, double threshold);

    void addition(double[][] matrix, double value);

    double[][] getFullIdentityMatrix(int column, int row);

    double[][] sum(double[][] delta, double[][] tmpE);

    double[] sum(double[] el1, double[] el2);
}
