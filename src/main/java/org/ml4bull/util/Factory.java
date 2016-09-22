package org.ml4bull.util;

import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.matrix.MatrixOperationsSimple;

public abstract class Factory {
    private static MatrixOperations matrixOperations;

    public static MatrixOperations getMatrixOperations() {
        if (matrixOperations == null) {
            matrixOperations = new MatrixOperationsSimple();
        }
        return matrixOperations;
    }

}
