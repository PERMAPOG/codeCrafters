from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('cowboy/', views.cowboy_filter_view, name='cowboy'), 
    path('apply_filter/', views.apply_filter_view, name='apply_filter'),
    path('face_mesh/', views.face_mesh_view, name='face_mesh'),
    path('face_mesh_page/', views.face_mesh_page_view, name='face_mesh_page'),
    path('upload/', views.face_mesh_view, name='upload'), 
    path('result/<str:img_url>/', views.result_view, name='result'),
]
