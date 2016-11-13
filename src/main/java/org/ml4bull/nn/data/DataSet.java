package org.ml4bull.nn.data;

import lombok.NonNull;

@lombok.Data
public class DataSet {
    @NonNull
    private double[][] input;
    @NonNull
    private double[][] output;
}
