from search import Search
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Поиск по корпусу',
        description='Осуществляется поиск по корпусу на основе романа "Пикник на обочине" Стругацких'
                    ' с помощью нескольких методов, указанных в начале работы программы.')
    parser.add_argument('-bm25', '--bm25', action='store_true',
                        help='добавить возможность поиска по значениям BM25')
    parser.add_argument('-w2v', '--word2vec', action='store_true',
                        help='добавить возможность поиска с помощью векторов Word2Vec')
    parser.add_argument('-ft', '--fasttext', action='store_true',
                        help='добавить возможность поиска через FastText')
    parser.add_argument('-ai', '--ai', action='store_true',
                        help='добавить возможность поиска через эмбеддинги модели')
    parser.add_argument('-n', '--num', action='store',
                        help='задать число результатов в выдаче (5 по умолчанию)')
    args = parser.parse_args()
    if args.num is None:
        num = 5
    else:
        if args.num.isnumeric():
            num = int(args.num)
        else:
            print('Неверный формат номера. Выставлено значение по умолчанию.')
            num = 5

    if not(args.bm25 or args.word2vec or args.fasttext or args.ai):
        print('Вы не выбрали ни одного метода. Завершение работы программы...')

    else:
        searcher = Search('./book.txt', bm25=args.bm25, w2v=args.word2vec, ft=args.fasttext,
                          ai=args.ai)
        inp_text = []
        methds = []
        if args.bm25:
            inp_text.append('-bm25 для поиска по значению BM25')
            methds.append('bm25')
        if args.word2vec:
            inp_text.append('-word2vec для использования значений векторов word2vec')
            methds.append('word2vec')
        if args.fasttext:
            inp_text.append('-fasttext для использования значений векторов fasttext')
            methds.append('fasttext')
        if args.ai:
            inp_text.append('-ai для использования значений векторов модели')
            methds.append('ai')

        while __name__ == '__main__':
            inp = input(f'Введите запрос в двойных кавычках и метод поиска ({", ".join(inp_text)}). '
                        f'Чтобы завершить работу программы, отправьте команду -end:\n')
            if inp == '-end':
                print('\nЗавершение работы программы.\n')
                break
            vals = inp.split('"')
            if len(vals) != 3:
                print('\nНеверный формат входных данных.\n')
                continue
            query = vals[1]
            method = vals[2].strip('- ')
            if method not in methds:
                print('\nНеверный формат метода.\n')
                continue

            print('', *searcher.gettop(query, method, num)[0], sep='\n')
