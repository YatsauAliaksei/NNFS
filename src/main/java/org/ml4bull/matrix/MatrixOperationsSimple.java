package org.ml4bull.matrix;

import com.google.common.base.Preconditions;
import org.jetbrains.annotations.NotNull;

public class MatrixOperationsSimple implements MatrixOperations {

    @Override
    public double[][] transpose(double[][] matrix) {
        double[][] transM = new double[matrix[0].length][matrix.length];

        ((DoubleIterator) (l, e) -> transM[e][l] = matrix[l][e]).iterate(matrix);

        return transM;
    }

    @Override
    public double[][] transpose(@NotNull double[] matrix) {
        double[][] transM = new double[1][matrix.length];
        System.arraycopy(matrix, 0, transM[0], 0, matrix.length);

        return transM;
    }

    @Override
    public void scalarMultiply(double[][] matrix, double k) {
        ((DoubleIterator) (l, e) -> matrix[l][e] *= k).iterate(matrix);
    }

    @Override
    public void scalarMultiply(double[] matrix, double k) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] *= k;
        }
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
    public double[] scalarMultiply(double[] matrix1, double[] matrix2) {
        if (matrix1.length != matrix2.length)
            throw new IllegalArgumentException("Row number of first matrix should be equal to column number of second matrix.");

        double[] result = new double[matrix1.length];
        for (int i = 0; i < matrix1.length; i++) {
            result[i] = matrix1[i] * matrix2[i];
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

    /**
     * @param matrix1 - T matrix
     * @param matrix2
     * @return
     */
    @Override
    public double[] multiplySingleDim(double[][] matrix1, double[] matrix2) {
        if (matrix1[0].length != matrix2.length)
            throw new IllegalArgumentException("Row number of first matrix should be equal to column number of second matrix.");

        double[] result = new double[matrix1.length];

        for (int i = 0; i < result.length; i++) {
            for (int l = 0; l < matrix1[i].length; l++) {
                result[i] += matrix1[i][l] * matrix2[l];
            }
        }

        return result;
    }

    @Override
    public void roundMatrix(double[] matrix, double threshold) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = matrix[i] >= threshold ? 1 : 0;
        }
    }

    @Override
    public void addition(double[][] matrix, double value) {
        ((DoubleIterator) (l, e) -> matrix[l][e] += value).iterate(matrix);
    }

    @Override
    public double[][] getFullIdentityMatrix(int column, int row) {
        double[][] matrix = new double[row][column];

        ((DoubleIterator) (l, e) -> matrix[l][e] = 1).iterate(matrix);

        return matrix;
    }

    @Override
    public double[][] sum(double[][] delta, double[][] tmpE) {
        if (delta == null) return tmpE;

        ((DoubleIterator) (l, e) -> delta[l][e] += tmpE[l][e]).iterate(delta);

        return delta;
    }

    @Override
    public double[] sum(double[] el1, double[] el2) {
        Preconditions.checkArgument(el1.length == el2.length);
        double[] result = new double[el1.length];

        for (int i = 0; i < el1.length; i++) {
            result[i] = el1[i] + el2[i];
        }

        return result;
    }

    @Override
    public boolean same(double[] i1, double[] i2) {
        for (int i = 0; i < i1.length; i++) {
            if (Double.compare(i1[i], i2[i]) != 0) {
                return false;
            }
        }
        return true;
    }

    @Override
    public double[] concatenate(double[] m1, double[] m2) {
        double[] result = new double[m1.length + m2.length];
        System.arraycopy(m1, 0, result, 0, m1.length);
        System.arraycopy(m2, 0, result, m1.length, m2.length);
        return result;
    }

}
