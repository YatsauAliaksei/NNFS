package org.ml4bull.util;

import lombok.Getter;

public class CharUtils {

    public static char vectorToChar(double[] v, Language language) {
        int maxIndex = 0;
        for (int i = 0; i < v.length; i++) {
            if (v[i] > v[maxIndex])
                maxIndex = i;
        }

        double[] predictedV = new double[language.charNum];
        predictedV[maxIndex] = 1;

        if (maxIndex == 0)
            return ' ';

        return (char) (MLUtils.transformClassToInt(predictedV) + language.leading);
    }

    public static double[] charToVector(char c, Language language) {
        return MLUtils.transformIntToClass(c == ' ' ? language.charNum : c - language.leading, language.charNum);
    }

    public static double[][] wordToVector(String word, Language language) {
        char[] chars = word.toCharArray();
        double[][] result = new double[chars.length + 1][];
        for (int i = 0; i < chars.length; i++) {
            result[i] = charToVector(chars[i], language);
        }
        result[result.length - 1] = MLUtils.transformIntToClass(language.charNum, language.charNum); // space
        return result;
    }

    public enum Language {
        RUSSIAN(33, 1071),
        US(27, 96);

        @Getter
        private int charNum, leading;

        Language(int charNum, int leading) {
            this.charNum = charNum; // plus space
            this.leading = leading;
        }
    }
}
