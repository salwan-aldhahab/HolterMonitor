from django.shortcuts import render

def ecg_view(request):
    return render(request, "ecg.html")