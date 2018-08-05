package org.ml4bull.nn.data;

@lombok.Data
public class Data {
    private double[] input;  // features
    private double[] output; // class. For regression use output[0]
}
