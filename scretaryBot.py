from itertools import groupby
import random
from collections import Counter
import numpy as np
import telebot
import matplotlib.pyplot as plt
import config
import todo_bd
import pickle
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import re
import numpy as np
from fuzzywuzzy import fuzz as f
from fuzzywuzzy import process as p
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words("russian")
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
morph = pymorphy2.MorphAnalyzer()
from sklearn.model_selection import train_test_split
import time
import schedule
from multiprocessing.context import Process



def lemm(message):
    """
    функция лемматизации для приведения слов к начальной форме
    """
    filtered_tokens = []
    st = ""
    tokens = word_tokenize(message.lower(), language="russian")
    for token in tokens:
        if token not in stop_words:
            token = morph.parse(token)[0].normal_form
            filtered_tokens.append(token)
            st += token + " "
    return re.sub(r'[^\w\s]', '', st)


data = pd.read_csv('pyTelegramBotAPI/s.csv', sep=';')
y = data['rubric'].values
sentences = data['message'].values
sentences = [lemm(item) for item in sentences]

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=1000)

vectorizer = TfidfVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

with open('pyTelegramBotAPI/text_classifier', 'wb') as picklefile:
    pickle.dump(classifier, picklefile)

with open('pyTelegramBotAPI/text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

bot = telebot.TeleBot(config.token)  # Подключение к боту


def train(sent, y_t):
    """метод для тренировки машинного обучения"""
    sentences_train, sentences_test, y_train, y_test = train_test_split(sent, y_t, test_size=0.2, random_state=1000)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    print(score)
    with open('pyTelegramBotAPI/text_classifier', 'wb') as picklefile:
        pickle.dump(classifier, picklefile)


# train(sentences, y)#####################################


@bot.message_handler(commands=['start', 'help'])
def start_msg(message):
    """
    start_msg - отвечает на команды 'start', 'help'
    :param message: текст команды
    :return:
    """
    if message.text == "/help" or message.text =="/help@LittleSecyBot":
        bot.send_message(message.chat.id, "Чтобы добавить задание, просто напиши его в чат умоминая того, кому оно адресовано. \n"
                                          "Для просмотра списка задач используй команду /tasklist. \n"
                                          "Чтобы просмотреть график сообщений введите команду /statistic. "
                                          "Чтобы посмотреть свои задачи введите команду /mytasks")
    else:
        bot.send_message(message.chat.id, "Данный бот позволяет составить список дел и отмечать в нем "
                                          "выполненные пункты.\nЧтобы просмотреть все доступные команды "
                                          "используйте /help ")


@bot.message_handler(commands=['tasklist'])
def print_alltasks(message):
    '''
    метод для бота, который выводит все активные задания для команды
    '''
    text = todo_bd.print_tasks(message.chat.id)
    if text == '':
        bot.reply_to(message, 'список дел на сегодня чист, все могут идти отдыхать')
    else:
        bot.send_message(message.chat.id, todo_bd.print_tasks(message.chat.id))


@bot.message_handler(commands=['mytasks'])
def print_tasklist(message):
    """
    print_tasklist - отправляет пользователю список дел из бд по id пользователь
    :param message: текст команды
    :return:
    """
    print(todo_bd.print_all())
    print(todo_bd.read_data_in_task(message.chat.id, '@' + message.from_user.username))
    # flag = True #todo_bd.check_tz(message.chat.id, message.from_user.username)
    # if flag:
    text = todo_bd.read_data_in_task(message.chat.id, '@' + message.from_user.username)
    if text == '':
        bot.reply_to(message, 'список дел на сегодня чист, можно отдыхать')
    else:
        bot.reply_to(message, text)


def load_all(chatid):
    """метод для получения категории сообщений"""
    x = todo_bd.get_tag_mess(chatid)
    return x


@bot.message_handler(commands=['statistic'])
def send_hist(message):
    """
    send_hist - отправляет гистограмму с данными о темах диалога
    :param message: текст команды
    :return:
    """
    plt.clf()
    data = load_all(message.chat.id)
    print(data)
    Xnew = np.array(data)
    Xnew = [lemm(item) for item in Xnew]
    X = vectorizer.transform(Xnew)
    ynew = model.predict(X)
    unique, pos = np.unique(ynew, return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    cnt = Counter()
    for word in ynew:
        cnt[word] += 1

    all_colors = list(plt.cm.colors.cnames.keys())
    random.seed(100)
    c = random.choices(all_colors, k=3)

    plt.figure(figsize=(16, 10), dpi=80)

    plt.bar(list(cnt.keys()), list(cnt.values()), color=c, width=.8)

    plt.gca().set_xticklabels(list(cnt.keys()), horizontalalignment='center', fontsize=20)
    plt.title("Сегодня вы общались на эти темы :)", fontsize=22)
    plt.ylim(0, counts[maxpos])
    plt.savefig('pyTelegramBotAPI/img/saved_figure.png')
    # todo_bd.delete_tb(message.chat.id)
    bot.send_photo(message.chat.id, open('pyTelegramBotAPI/img/saved_figure.png', 'rb'))


@bot.message_handler(content_types=['text'])
def get_task(message):
    """
    get_task - добавляет новый пункт к списку
    :param message: сообщение пользователя
    :return:
    """
    text = message.text.replace("\n", " ")
    todo_bd.add_mess(message.chat.id, text)

    lines = str(message.text).split("\n")

    text = todo_bd.read_data_in_task(message.chat.id, '@' + message.from_user.username)
    text = text.split('\n')

    for i in text:
        a = f.token_sort_ratio(message.text, i)
        print(a)
        if a >= 70:
            todo_bd.delete_task(message.chat.id, i)
            print(a)
        else:
            pass

    for i in lines:
        if "@" in i:
            rez = i
            words = rez.split(' ')
            fragment = '@'
            new_words = []
            name = []
            rez = []
            for word in words:
                if fragment not in word:
                    new_words.append(word)
                if word.startswith('@'):
                    name.append(word)
                    rez.append(word)
            s = ' '.join(new_words)
            rez.append(s)
            print(str(new_words))
            print(name)
            print(rez)
            for x in name:
                # x = x.replace("@", "")
                todo_bd.add_task(message.chat.id, str(x), s)
    # print(todo_bd.print_tasks(message.chat.id))


def clean_bd():
    """метод для очистки базы данных"""
    #todo_bd.delete_todo()
    todo_bd.delete_mess()


schedule.every().day.at("14:22").do(clean_bd)


class ScheduleMessage():
    """класс для создания расписания"""
    def try_send_schedule():
        """свой цикл для запуска планировщика с периодом в 1 секунду:"""
        while True:
            schedule.run_pending()
            time.sleep(1)

    def start_process():
        """запуск процесса на считывание локального времени на сервере"""
        p1 = Process(target=ScheduleMessage.try_send_schedule, args=())
        p1.start()


if __name__ == '__main__':  # Ожидать входящие сообщения
    todo_bd.init_db()
    ScheduleMessage.start_process()
    bot.infinity_polling()