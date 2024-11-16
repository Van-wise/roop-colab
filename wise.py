# -- 下载模型 26s
import os
import sys
import subprocess
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor
from IPython.display import clear_output, display, HTML

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

# -- 修复degradations 3s
def fix():
    full_version = sys.version.split(' ')[0]
    major_minor_version = '.'.join(full_version.split('.')[:2])
    basicsr_path = f"/usr/local/lib/python{major_minor_version}/dist-packages/basicsr/data/degradations.py"
    local_path = "/content/roop/degradations.py"
    if os.path.exists(local_path):
        try:
            subprocess.run(["cp", local_path, basicsr_path], check=True)
            print(f"Copied to {basicsr_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during copy: {e}")
            print("Check the command and file permissions.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    else:
        print(f"Local file {local_path} not found.")

# -- 安装依赖 25s
def install_dependencies():
    for cmd in [
        'pip install --progress-bar off --quiet /content/roop/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl',
        'pip install --progress-bar off --quiet onnx==1.14.0 insightface==0.7.3 tk==0.1.0 customtkinter==5.2.0 gfpgan==1.3.8 protobuf==3.20.3',
        'pip install --progress-bar off --quiet --no-cache-dir -I tkinterdnd2-universal==1.7.3 tkinterdnd2==0.3.0'
    ]:
        result = subprocess.run(cmd, shell=True)
        print(f"{' '.join(cmd.split()[5:])} installed successfully." if result.returncode == 0 else "")

# -- 手机保持运行 1s
def mobile_keepalive(opt):
    if str(opt) == "True":
        html_code = f'<audio src="https://raw.githubusercontent.com/KoboldAI/KoboldAI-Client/main/colab/silence.m4a" autoplay controls muted></audio>'
        display(HTML(html_code))
        
# -- 挂载云盘 15s
def content_models(link_google_drive):
    try:
        if os.path.exists('/content/drive'):
            print('谷歌云盘已挂载...')
        elif link_google_drive:
            from google.colab import drive
            drive.mount('/content/drive')
            print('Google Drive 挂载成功！')
        else:
            print('暂时不挂载谷歌云盘...')
    except Exception as e:
        print(f"An error occurred: {e}")
        
# -- 确定素材路径 20s
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab import files
from PIL import Image
from urllib.parse import urlparse
from pathlib import Path
import moviepy.editor as mp
from base64 import b64encode
from IPython.display import HTML, display

def clean_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def generate_unique_path(base_path):
    counter = 1
    new_path = base_path
    while new_path.exists():
        new_path = base_path.with_stem(f"{base_path.stem}_{counter}")
        counter += 1
    return new_path

def download_media(url, target_folder, max_file_size=100 * 1024 * 1024, max_retries=3):
    for attempt in range(max_retries):
        try:
            cleaned = clean_url(url)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': urlparse(cleaned).netloc,
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            head_response = requests.head(cleaned, headers=headers, allow_redirects=True)
            head_response.raise_for_status()
            content_length = int(head_response.headers.get('Content-Length', 0))
            if content_length > max_file_size:
                file_size_mb = content_length / (1024 * 1024)
                raise ValueError(f"文件过大！当前文件大小: {file_size_mb:.2f} MB，最大允许大小: {max_file_size / (1024 * 1024)} MB。建议：1. 使用网盘分享链接；2. 压缩文件；3. 选择更小的媒体文件。")
            response = requests.get(cleaned, headers=headers, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            media_path = Path(target_folder) / Path(cleaned).name
            media_path.parent.mkdir(parents=True, exist_ok=True)
            with open(media_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            return media_path
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"下载媒体文件失败: {e}") from e

def get_local_media(source):
    if not os.path.exists(source):
        raise FileNotFoundError(f"文件 {source} 不存在！")
    return Path(source)

def get_media(source, save_to_path=1, max_file_size=100 * 1024 * 1024, max_retries=3): 
    if not source:
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("用户已取消上传！")
        return Path('/content') / next(iter(uploaded.keys()))
    if source.startswith('/content/'):
        return get_local_media(source)
    if save_to_path == 1:
        return download_media(source, '/content/source', max_file_size, max_retries)
    elif save_to_path == 2:
        return download_media(source, '/content/target', max_file_size, max_retries)
    else:
        raise ValueError("save_to_path 参数的值必须为 1 或 2！")

def convert_image_format(media_path, target_folder):
    if media_path.suffix.lower()!= '.jpg':
        new_path = generate_unique_path(Path(target_folder) / f"{media_path.stem}.jpg")
        try:
            img = Image.open(media_path).convert('RGB')
            img.save(new_path, quality=95)
            return new_path
        except Exception as e:
            raise RuntimeError(f"图片格式转换失败: {e}") from e
    return media_path

def convert_video_format(media_path, target_folder):
    if media_path.suffix.lower() == '.mp4':
        return media_path
    new_path = generate_unique_path(Path(target_folder) / f"{media_path.stem}.mp4")
    try:
        video = mp.VideoFileClip(str(media_path))
        video.write_videofile(str(new_path), codec='libx264')
        video.close()
        return new_path
    except Exception as e:
        raise RuntimeError(f"视频格式转换失败: {e}") from e

def convert_media_format(media_path, target_folder, image_extensions=('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.webp'), video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
    if not media_path:
        raise ValueError("媒体文件不存在！")
    if media_path.suffix.lower() in image_extensions:
        return convert_image_format(media_path, target_folder)
    elif media_path.suffix.lower() in video_extensions:
        return convert_video_format(media_path, target_folder)
    raise ValueError(f"不支持的文件格式: {media_path}")

def display_image(media_path):
    plt.figure(figsize=(4, 3)) 
    plt.imshow(mpimg.imread(media_path))
    plt.axis('off')
    plt.show()

def display_video(media_path, preview_duration=10):
    preview_path = str(media_path).replace('.mp4', '_preview.mp4')
    try:
        subprocess.run([
            'ffmpeg',
            '-i', str(media_path),
            '-t', str(preview_duration),
            '-c', 'copy',
            preview_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        subprocess.run([
            'ffmpeg',
            '-i', str(media_path),
            '-t', str(preview_duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            preview_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(preview_path, 'rb') as f:
        video_data = f.read()
    os.remove(preview_path)
    data_url = "data:video/mp4;base64," + b64encode(video_data).decode()
    display(HTML(f'''
    <video width=300 height=200 controls>  
        <source src="{data_url}" type="video/mp4">
    </video>
    '''))

def display_media(source, show_media=True, save_to_path=1, preview_duration=10):  
    target_folder = Path("/content/target")
    target_folder.mkdir(exist_ok=True)
    try:
        media_path = get_media(source, save_to_path) 
        media_path = convert_media_format(media_path, target_folder)
        if show_media:  
            if media_path.suffix.lower() in ('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.webp'):
                display_image(media_path)
            elif media_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                display_video(media_path, preview_duration)
        return str(media_path)
    except Exception as e:
        print(e)
        return None

# For URL sources
url11 = "/content/5e5d002541ec5.vid"  #@param {type:"string"}
show_media = True # @param {"type":"boolean"}
save_to_path = 2 # @param {"type":"integer","placeholder":"1"}
result_path = display_media(url11, show_media, save_to_path)

if result_path:
  print(f"当前人脸素材路径为: {result_path}")

# -- star
download_all_models(models_info)
install_dependencies()
fix()