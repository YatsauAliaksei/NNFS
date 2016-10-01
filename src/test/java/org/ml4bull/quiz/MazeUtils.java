package org.ml4bull.quiz;

import java.util.Random;

public class MazeUtils {

    public static int[] checkNeighbor(String[][] array, int x, int y, int dx, int dy) {
        if (x + dx >= 0 && x + dx <= array.length - 1 && y + dy >= 0 && y + dy <= array.length - 1) {
            String value = array[x + dx][y + dy];
            return value.equals("X") ? null : new int[]{x + dx, y + dy};
        } else {
            return null;
        }
    }

    public static boolean isUnreachable(String[][] array) {
        if (array[0][0] != "X" && array[9][9] != "X") {
            int[] arr = new int[array.length * array[0].length];
            arr[0] = 1;
            virus(0, 0, array, arr);
            return arr[0] != arr[arr.length - 1];
        } else {
            return true;
        }
    }

    public static void virus(int line, int row, String[][] array, int[] arr) {
        if (!array[line][row].equals("X")) {
            int[][] var4 = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
            int var5 = var4.length;

            for (int var6 = 0; var6 < var5; ++var6) {
                int[] neighbor = var4[var6];
                int[] nInd = checkNeighbor(array, line, row, neighbor[0], neighbor[1]);
                if (nInd != null) {
                    int i = nInd[0] * array.length + nInd[1];
                    if (arr[i] == 0) {
                        arr[i] = 1;
                        virus(nInd[0], nInd[1], array, arr);
                    }
                }
            }

        }
    }

    public static String[][] generateMaze(int x, int y) {
        String[][] ar = new String[x][y];
        Random random = new Random();

        for (int i = 0; i < ar.length; ++i) {
            for (int i1 = 0; i1 < ar[i].length; ++i1) {
                int value = random.nextInt(2);
                if (value == 1) {
                    value = random.nextInt(2);
                    ar[i][i1] = value == 1 ? "X" : "-";
                } else {
                    ar[i][i1] = "-";
                }
            }
        }

        return ar;
    }

    public static void mazePrint(String[][] ar, String separator) {
        String[][] var2 = ar;
        int var3 = ar.length;

        for (int var4 = 0; var4 < var3; ++var4) {
            String[] anAr = var2[var4];
            String[] var6 = anAr;
            int var7 = anAr.length;

            for (int var8 = 0; var8 < var7; ++var8) {
                String i2 = var6[var8];
                String delimiter = i2.length() > 1 ? separator.substring(1) : separator;
                System.out.print(i2 + delimiter);
            }

            System.out.println();
        }

    }
}
