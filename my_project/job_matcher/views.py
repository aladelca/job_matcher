import pickle

import pandas as pd
from django.shortcuts import redirect, render
from sklearn.metrics.pairwise import cosine_similarity

from .forms import DocumentForm
from .models import Document
from .preprocess import get_text, preprocess_text

# Create your views here.


def index(request):
    return render(request, "index.html")


def process_cv(request):
    if request.method == "POST":
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect("index")
    else:
        form = DocumentForm()
    return render(request, "process_cv.html", {"form": form})


def list_cv(request):
    df = pd.DataFrame(list(Document.objects.all().values()))
    df["id"] = df["id"].astype(str)

    def add_url(data):
        return f"<a href='{data}'>{data}</a>"

    df["id"] = df["id"].apply(add_url)

    html_table = df.to_html(
        escape=False,
        index=False,
        border=1,
        classes="table table-striped table-hover",
    )
    return render(request, "list_cv.html", {"html_table": html_table})


def preprocess_cv(request, cv_id):
    filename = Document.objects.values_list("document", flat=True).get(id=cv_id)
    cv_text = get_text(filename)
    df = pd.DataFrame({"text": [cv_text]})
    df = preprocess_text(df, "text")
    df["text"] = df["text"].apply(lambda x: " ".join(x))
    # Load vect and jobs

    vect = pickle.load(open("job_matcher/static/data/job_vectorizer.pickle", "rb"))
    jobs = pickle.load(open("job_matcher/static/data/puestos.pickle", "rb"))

    # Preprocess the jobs
    jobs = preprocess_text(jobs, "PUESTO")
    jobs["PUESTO"] = jobs["PUESTO"].apply(lambda x: " ".join(x))
    # Get the vector for the cv
    cv_vector = vect.transform(df["text"])
    jobs_vector = vect.transform(jobs["PUESTO"])

    # Calculate the cosine similarity

    similarity = cosine_similarity(cv_vector, jobs_vector)
    jobs["similarity"] = similarity[0]
    jobs = jobs.sort_values(by="similarity", ascending=False).head(5)
    jobs = jobs[["PUESTO", "similarity"]]

    html_ranking = jobs.to_html(
        escape=False,
        index=False,
        border=1,
        classes="table table-striped table-hover",
    )

    return render(request, "match.html", {"html_match": html_ranking})
