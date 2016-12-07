package org.ml4bull.data;

import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.stream.Collectors;

@Slf4j
public class WebBandog {

    public Set<String> findWords(String startPage, int count, int maxCharLength) {
        final Set<String> words = new HashSet<>(count + 1, 1);
        Deque<String> queue = new ConcurrentLinkedDeque<>();
        queue.add(startPage);

        while (words.size() < count) {
            log.info("Size: {}", words.size());

            final Document doc;
            try {
                doc = collectWords(queue.pop(), words);
            } catch (Exception e) {
                log.error(e.getMessage());
                continue;
            }

            words.addAll(words.parallelStream()
                    .filter(w -> w.trim().length() <= maxCharLength && w.trim().length() > 2)
                    .collect(Collectors.toSet()));

            // get all links
            Elements links = getLinks(doc);
//            printLinks(links);
            Set<String> s = links.stream().map(l -> l.attr("abs:href")).collect(Collectors.toSet());

            queue.addAll(s);
        }

        return words;
    }

    private Document collectWords(String pageURL, Set<String> words) {
        System.out.println("Open page: " + pageURL);
        Document doc = getDocument(pageURL);
        Element body = doc.body();
        if (body == null)
            throw new RuntimeException("No body");

        String pageText = body.text();

        words.addAll(getWords(pageText));
        return doc;
    }

    @NotNull
    private Set<String> getWords(String pageText) {
        String[] s = pageText.replaceAll("[^a-zA-Z]", " ").replaceAll("\\s+", " ").toLowerCase().trim().split(" ");
        return Sets.newHashSet(s);
    }

    private Elements getLinks(Document doc) {return doc.select("a[href]");}

    private Document getDocument(String pageURL) {
        String userAgent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:50.0) Gecko/20100101 Firefox/50.0";

        try {
            return Jsoup.connect(pageURL).userAgent(userAgent).get();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void print(String msg, Object... args) {
        System.out.println(String.format(msg, args));
    }

    private String trim(String s, int width) {
        if (s.length() > width)
            return s.substring(0, width - 1) + ".";
        else
            return s;
    }

    private void printLinks(Elements links) {
        print("\nLinks: (%d)", links.size());
        for (Element link : links) {
            print(" * a: <%s>  (%s)", link.attr("abs:href"), trim(link.text(), 35));
        }
    }
}
