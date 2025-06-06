from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from ultralytics import YOLO
import os
import shutil

# Load YOLO model
model = YOLO("yolov8n.pt")  # or your custom model

def detect_objects(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_image_path = fs.path(filename)
        uploaded_image_url = fs.url(filename)

        try:
            # Run YOLO detection and save result
            model.predict(source=uploaded_image_path, save=True)

            # Find the latest result folder
            result_dir = os.path.join("runs", "detect")
            latest_folder = sorted(os.listdir(result_dir))[-1]
            result_file_path = os.path.join(result_dir, latest_folder, filename)

            # Copy result image to MEDIA folder to serve it in template
            result_media_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
            os.makedirs(os.path.dirname(result_media_path), exist_ok=True)
            shutil.copy(result_file_path, result_media_path)

            # Create a URL to access the result image
            result_url = os.path.join(settings.MEDIA_URL, 'results', filename)

            context = {
                'uploaded_image': uploaded_image_url,
                'result_image': result_url,
            }

        except Exception as e:
            context['error'] = f"Error during detection: {e}"

    return render(request, 'detect.html', context)
