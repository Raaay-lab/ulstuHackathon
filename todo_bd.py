import sqlite3


def init_db():
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    # Если таблицы не существует создать ее
    cursor.execute("""CREATE TABLE IF NOT EXISTS 'todo'(id TEXT, user TEXT, teg TEXT)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS 'mess' (id TEXT, teg TEXT)""")
    conn.commit()


def check_tz(chatid, name): # проверка наличия элементов в бд
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    cursor.execute(f"""SELECT teg FROM 'todo' WHERE id={chatid} AND user={name}""")
    row = cursor.fetchone()
    if row is None:
        return False
    return True

def delete():
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    cursor.execute("""DROP DATABASE""")

def check_mess(chatid): # проверка наличия элементов в бд
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    cursor.execute(f"""SELECT teg FROM 'mess' WHERE id={chatid}""")
    row = cursor.fetchone()
    if row is None:
        return False
    return True


def get_tag_tz(chatid): # выводит категории заданий
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    c = cursor.execute(f"""SELECT teg FROM 'todo' WHERE id={chatid}""")
    result = '\n'.join(['| '.join(map(str, x)) for x in c])
    result = result.split('\n')
    return result


def get_tag_mess(chatid): # выводит категори сообщений
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    c = cursor.execute(f"""SELECT teg FROM 'mess' WHERE id={chatid}""")
    result = '\n'.join(['| '.join(map(str, x)) for x in c])
    result = result.split('\n')
    return result


def get_tz_by_tag(chatid, teg): # выводит хронологию сообщений задания по тегу
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    c = cursor.execute(f"""SELECT mess FROM 'todo' WHERE id={chatid} AND teg={teg}""")
    result = '\n'.join(['| '.join(map(str, x)) for x in c])
    result = result.split('\n')
    return result


def add_task(chatid, name, teg):  # Функция добавляет данные в таблицу
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    ins = f"""INSERT INTO 'todo'  VALUES ('{chatid}', '{name}', '{teg}')"""
    cursor.execute(ins)
    conn.commit()


def add_mess(chatid, teg):  # Функция добавляет данные в таблицу
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    ins = f"""INSERT INTO 'mess'  VALUES ('{chatid}', '{teg}')"""
    cursor.execute(ins)
    conn.commit()

def print_tasks(chatid):
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    c = cursor.execute(f"""SELECT  user,teg FROM 'todo' WHERE id={chatid}""")
    result1 = '\n'.join(['| '.join(map(str, x)) for x in c])
    '''cursor = conn.cursor()
    t = cursor.execute("""SELECT teg FROM 'todo' WHERE id=(?)""", (chatid))
    result2 = '\n'.join(['| '.join(map(str, x)) for x in t])'''
    return result1 #+ result2


def read_data_in_task(chatid, name):  # Чтение данных из таблицы
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    c = cursor.execute("""SELECT teg FROM 'todo' WHERE id=(?) AND user=(?)""", (chatid, name))
    result = '\n'.join(['| '.join(map(str, x)) for x in c])
    return result

def print_all():
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    c = cursor.execute(f"""SELECT * FROM 'todo'""")
    result = '\n'.join(['| '.join(map(str, x)) for x in c])
    return result


def delete_task(chatid, text):  # Удаление данных из таблицы
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    delete = f"""DELETE FROM 'todo' WHERE id = '{chatid}' AND teg = '{text}' """
    cursor.execute(delete)
    conn.commit()


def delete_todo(): # удаляет все записи из таблицы сообщений
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    cursor.execute(f"""DELETE FROM 'todo'""")
    conn.commit()


def delete_mess(): # удаляет все записи из таблицы сообщений
    conn = sqlite3.connect("db.db")
    cursor = conn.cursor()
    cursor.execute(f"""DELETE FROM 'mess'""")
    conn.commit()