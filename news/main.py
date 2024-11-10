from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from googletrans import Translator

app = Flask(__name__)



def fetch_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_content = ' '.join([para.get_text() for para in paragraphs])
        if len(article_content) == 0:
            raise ValueError("Could not extract article content.")
        return article_content
    except Exception as e:
        print(f"Error fetching article content: {e}")
        return None



def split_into_chunks(text, chunk_size=1024):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]



def summarize_long_article(article_content, max_words=150):
    chunks = split_into_chunks(article_content)
    summaries = []

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    try:
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_words, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        full_summary = ' '.join(summaries)

        return ' '.join(full_summary.split()[:max_words])
    except Exception as e:
        print(f"Summarization failed: {e}")
        return None


def translate_summary(summary, target_language):
    translator = Translator()
    try:
        translated_summary = translator.translate(summary, dest=target_language).text
        return translated_summary
    except Exception as e:
        print(f"Translation failed: {e}")
        return summary



@app.route('/', methods=['GET', 'POST'])
def home():
    summary = None
    if request.method == 'POST':
        url = request.form['url']
        max_words = int(request.form['max_words'])
        target_language = request.form['language']

        article_content = fetch_article_content(url)
        if article_content:
            summary = summarize_long_article(article_content, max_words=max_words)
            if target_language != 'none':
                summary = translate_summary(summary, target_language)

    return render_template('index.html', summary=summary)


if __name__ == '__main__':
    app.run(debug=True)