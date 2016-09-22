package org.ml4bull.matrix;

public class MatrixOperationsSimple implements MatrixOperations {

    @Override
    public double[][] transpose(double[][] matrix) {
        double[][] transM = new double[matrix[0].length][matrix.length];

        ((DoubleIterator) (l, e) -> transM[e][l] = matrix[l][e]).iterate(matrix);

        return transM;
    }

    @Override
    public double[][] transpose(double[] matrix) {
        double[][] transM = new double[1][matrix.length];
        System.arraycopy(matrix, 0, transM[0], 0, matrix.length);

        return transM;
    }

    @Override
    public void scalarMultiply(double[][] matrix, double k) {
        ((DoubleIterator) (l, e) -> matrix[l][e] *= k).iterate(matrix);
    }

    @Override
    public double multiply(double[] matrix1, double[] matrix2) {
        if (matrix1.length != matrix2.length)
            throw new IllegalArgumentException("Row number of first matrix should be equal to column number of second matrix.");

        double result = 0;
        for (int i = 0; i < matrix1.length; i++) {
            result += matrix1[i] * matrix2[i];
        }

        return result;
    }

    @Override
    public double[][] multiply(double[][] matrix1, double[][] matrix2) {
        if (matrix1[0].length != matrix2.length)
            throw new IllegalArgumentException("Row number of first matrix should be equal to column number of second matrix.");

        double[][] result = new double[matrix1.length][matrix2[0].length];
        ((DoubleIterator) (l, e) -> {
            for (int i = 0; i < matrix1[l].length; i++) {
                result[l][e] += matrix1[l][i] * matrix2[i][e];
            }

        }).iterate(result);

        return result;
    }

    public void addition(double[][] matrix, double value) {
        ((DoubleIterator) (l, e) -> matrix[l][e] += value).iterate(matrix);
    }

    public double[][] getFullIdentityMatrix(int column, int row) {
        double[][] matrix = new double[row][column];

        ((DoubleIterator) (l, e) -> matrix[l][e] = 1).iterate(matrix);

        return matrix;
    }
}
