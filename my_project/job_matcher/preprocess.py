import re

import docx
import pandas as pd
import stanza
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode

stanza.download("es")


def clean_text(texto):
    text_ascii = unidecode(texto)
    result = re.sub(r"[^A-Za-z0-9 ]", "", text_ascii)
    result = result.lower()
    return result


def get_tokens(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(lambda x: word_tokenize(x))
    return df


def remove_stopwords(
    df: pd.DataFrame, stopwords: list, tokens_col: str
) -> pd.DataFrame:
    df[tokens_col] = df[tokens_col].apply(
        lambda x: [word for word in x if word not in stopwords]
    )
    return df


def lemmatize_stanza(text_list):
    nlp = stanza.Pipeline("es")
    final = []
    for text in text_list:
        doc = nlp(text)
        for sentence in doc.sentences:
            for word in sentence.words:
                final.append(word.lemma)
    return final


def lemmatize_tokens(df: pd.DataFrame, tokens_col: str) -> pd.DataFrame:
    df[tokens_col] = df[tokens_col].apply(lambda x: lemmatize_stanza(x))
    return df


def get_text(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)


def preprocess_text(
    df: pd.DataFrame,
    text_col: str,
    stopwords: list = list(set(stopwords.words("spanish"))),
) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df = get_tokens(df, text_col)
    df = remove_stopwords(df, stopwords, text_col)
    df = lemmatize_tokens(df, text_col)
    return df
