package org.ml4bull.mnist;

import lombok.SneakyThrows;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.function.Function;
import java.util.function.Predicate;

public class DataSetLoader {

    @SneakyThrows(IOException.class)
    public static <T> T load(String filePath, Predicate<Integer> magicPredicate, Function<DataInputStream, T> function) {
        try (InputStream inputStream = Files.newInputStream(
                Paths.get(filePath), StandardOpenOption.READ);
             DataInputStream ois = new DataInputStream(inputStream)) {

            int magic = ois.readInt();
            if (!magicPredicate.test(magic))
                return null;

            return function.apply(ois);
        }
    }
}
