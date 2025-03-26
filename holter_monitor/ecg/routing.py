from django.urls import re_path
from .consumers import ECGConsumer

websocket_urlpatterns = [
    re_path(r'ws/ecg/$', ECGConsumer.as_asgi()),
]