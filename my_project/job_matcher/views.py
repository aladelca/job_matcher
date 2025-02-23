from django.shortcuts import render, redirect
from .forms import DocumentForm 
from .models import Document
import pandas as pd
# Create your views here.

def index(request):
    return render(request, 'index.html')

def process_cv(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else: 
        form = DocumentForm()
    return render(request, 'process_cv.html', {'form': form})

def list_cv(request):
    df = pd.DataFrame(list(Document.objects.all().values()))
    df["id"] = df["id"].astype(str)
    
    def add_url(data):
        return "<a href='" + data + "'>" + data + "</a>"
    
    df["id"] = df["id"].apply(add_url)

    html_table = df.to_html(
        escape=False,
        index=False,
        border = 1,
        classes= "table table-striped table-hover"
        )
    return render(request, 'list_cv.html', {'html_table': html_table})
    