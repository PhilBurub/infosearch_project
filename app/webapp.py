from flask import Flask, render_template, request
from search import Search

app = Flask(__name__)
searcher = Search('./book.txt', bm25=True, w2v=True, ft=False, ai=False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    return render_template('search.html')


@app.route('/results', methods=['get'])
def results():
    args = request.args
    num = int(args.get('num')) if args.get('num') != '' and args.get('num').isnumeric() else 5
    results = searcher.gettop(args.get('query'), args.get('methods'), num)
    time = results[-3]
    results = results[:-4]
    return render_template("results.html", results=results, time=time)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)