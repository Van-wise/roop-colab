# -- 下载模型 26s
import os
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor

models_info = [
    ('https://github.com/karaokenerds/python-audio-separator/releases/download/v0.12.1/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', 'onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', '/content/roop/'),
    ('https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx', 'inswapper_128.onnx', '/content/roop/checkpoints/'),
    ('https://github.com/Hillobar/Rope/releases/download/Sapphire/inswapper_128.fp16.onnx', 'inswapper_128.fp16.onnx', '/content/roop/checkpoints/'),
    ('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', 'buffalo_l.zip', '/content/'),
    ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', 'GFPGANv1.4.pth', '/content/roop/models/'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', 'detection_Resnet50_Final.pth', '/content/roop/gfpgan/weights/'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', 'parsing_parsenet.pth', '/content/roop/gfpgan/weights/')
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
            extract_zip(local_path,"/content/roop/checkpoints/models/buffalo_l")
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

# -- 修复degradations 3s
import sys
import shutil

def fix():
    full_version = sys.version.split(' ')[0]
    major_minor_version = '.'.join(full_version.split('.')[:2])
    basicsr_path = f"/usr/local/lib/python{major_minor_version}/dist-packages/basicsr/data/degradations.py"
    local_path = "/content/roop/degradations.py"
    if os.path.exists(local_path):
        try:
            shutil.copy(local_path, basicsr_path)
            print(f"Copied to {basicsr_path}")
        except Exception as e:
            print(f"An error occurred during copy: {e}")
            print("Check the paths and file permissions.")
    else:
        print(f"Local file {local_path} not found.")

#fix()

# -- 安装依赖 25s
def install_dependencies():
    package_info = [
        ('onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', '/content/roop/'),
        ('onnx==1.14.0, insightface==0.7.3, tk==0.1.0, customtkinter==5.2.0, gfpgan==1.3.8, protobuf==3.20.3', ' --no-cache-dir -I '),
        ('tkinterdnd2-universal==1.7.3, tkinterdnd2==0.3.0', ' --no-cache-dir -I ')
    ]
    for info in package_info:
        install_command = 'pip install --progress-bar off --quiet'+ info[1] + info[0]
        return_code = os.system(install_command)
        if return_code == 0:
            print(f"{info[0]} installed successfully.")
        else:
            print(f"Failed to install {info[0]}")

#install_dependencies()

download_all_models(models_info)
install_dependencies()
fix()