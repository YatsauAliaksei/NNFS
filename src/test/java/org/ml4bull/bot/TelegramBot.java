package org.ml4bull.bot;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import lombok.AccessLevel;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.telegram.telegrambots.ApiContextInitializer;
import org.telegram.telegrambots.TelegramBotsApi;
import org.telegram.telegrambots.api.methods.send.SendMessage;
import org.telegram.telegrambots.api.objects.Message;
import org.telegram.telegrambots.api.objects.Update;
import org.telegram.telegrambots.bots.DefaultBotOptions;
import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.exceptions.TelegramApiException;
import org.telegram.telegrambots.exceptions.TelegramApiRequestException;

import java.io.IOException;
import java.util.List;
import java.util.Properties;

@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
@Slf4j
public class TelegramBot extends TelegramLongPollingBot {

    public static long DEFAULT_CHAT_ID = -286755964L;
    @NonNull
    private String botUsername;
    @NonNull
    private String botToken;

    private TelegramBot(DefaultBotOptions options) {
        super(options);
    }

    @Override
    public void onUpdateReceived(Update update) {

    }

    @Override
    public String getBotUsername() {
        return botUsername;
    }

    @Override
    public String getBotToken() {
        return botToken;
    }

    @SneakyThrows
    public Message say(long chatId, String message) {
        return execute(new SendMessage(chatId, message));
    }

    @SneakyThrows
    public Message say(String message) {
        return execute(new SendMessage(DEFAULT_CHAT_ID, message));
    }

    @Override
    public void clearWebhook() throws TelegramApiRequestException {}

    private void registerBot() throws TelegramApiRequestException {
        TelegramBotsApi telegramBotsApi = new TelegramBotsApi();
        telegramBotsApi.registerBot(this);
        log.info("Bot {} has been registered successfully", botUsername);
    }

    private static TelegramBot instance;

    public static TelegramBot takeMe(String botUsername) {
        Preconditions.checkArgument(StringUtils.isNotBlank(botUsername), "Bot name mandatory field.");
        if (instance == null) { // no synchronization. Simplest version.
            Properties props = new Properties();
            try {
                props.load(TelegramBot.class.getResourceAsStream("/bots.exclude.properties"));
            } catch (IOException e) {
                e.printStackTrace();
            }
            List<String> botNames = Splitter.on(",").trimResults().splitToList(props.getProperty("bot.names"));
            String botName = botNames.stream()
                    .filter(name -> name.equalsIgnoreCase(botUsername))
                    .findFirst().orElseThrow(RuntimeException::new);

            ApiContextInitializer.init();
            instance = new TelegramBot(botName, props.getProperty(botName + ".token"));
            try {
                instance.registerBot();
            } catch (TelegramApiRequestException e) {
                throw new RuntimeException(e);
            }
        }
        return instance;
    }
}
