# -- 下载模型 25s
import os
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor

models_info = [
    ('https://github.com/karaokenerds/python-audio-separator/releases/download/v0.12.1/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', 'onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', '/content/roop-colab/'),
    ('https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx', 'inswapper_128.onnx', '/content/roop-colab/checkpoints/'),
    ('https://github.com/Hillobar/Rope/releases/download/Sapphire/inswapper_128.fp16.onnx', 'inswapper_128.fp16.onnx', '/content/roop-colab/checkpoints/'),
    ('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', 'buffalo_l.zip', '/content/'),
    ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', 'GFPGANv1.4.pth', '/content/roop-colab/models/'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', 'detection_Resnet50_Final.pth', '/content/roop-colab/gfpgan/weights/'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', 'parsing_parsenet.pth', '/content/roop-colab/gfpgan/weights/')
]

def download_model(url, name, path):
    local_path = os.path.join(path, name)
    try:
        if not os.path.exists(local_path):
            os.makedirs(path, exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
        print(f"{name} 下载成功!")
        if name == 'buffalo_l.zip':
            extract_zip(local_path,"/content/roop-colab/checkpoints/models/buffalo_l")
            print(f"{name} 解压成功!")
    except Exception as e:
        print(f"{name} 文件下载错误：{e}")

def download_all_models(models_info):
    with ThreadPoolExecutor(max_workers=10) as executor:
        for info in models_info:
            executor.submit(download_model, *info)

def extract_zip(zip_file_path, extract_path):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        print(f"解压 {zip_file_path} 错误: {e}")

#download_all_models(models_info)