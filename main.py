print('Hello')
from transformers import pipeline
from flask import Flask, redirect, url_for, request, render_template
import warnings

warnings.filterwarnings('ignore')

# Model Weight Load
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")


def summarize_text(text: str):
    summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
    print(summary_text)

    return summary_text


app = Flask(__name__)


def before_request():
    app.jinja_env.cache = {}


@app.route('/')
def welcome():
    return render_template('start.html')


@app.route('/summary', methods=['POST', 'GET'])
def summary():
    if request.method == 'POST':
        text = request.form['raw_text']
        print(text)

        text_summary = summarize_text(text)
        # return redirect(url_for('summary',text = text, summary=''))
        return render_template('index.html', text=text, summary=text_summary)
    else:
        text = ''
        return render_template('index.html', text=text, summary='')


if __name__ == '__main__':
    # app.before_request(before_request)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)
