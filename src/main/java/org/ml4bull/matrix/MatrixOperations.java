package org.ml4bull.matrix;

/**
 * Created by AYatsev.
 */
public interface MatrixOperations {

    double[][] transpose(double[][] matrix);

    void multiply(double[][] matrix, double k);

    default void printMatrix(double[][] matrix) {
        for (double[] m : matrix) {
            for (double m1 : m) {
                System.out.print(m1 + " ");
            }
            System.out.println();
        }
    }

    double[][] multiply(double[][] matrix1, double[][] matrix2);
}
