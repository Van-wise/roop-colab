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
import subprocess

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
        
# -- 确定人脸图片路径 3s
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab import files
from PIL import Image

source = "https://cdn.bncloudfl.com/bn/f62/b4e/976/f62b4e9764dc8773e43ebe6953f765d5c8909ef0.gif"  #@param {type:"string"}

def display_image(source):
    image_path = None  # 先初始化图片路径为None
    target_folder = "/content/source"  # 定义保存图片的文件夹路径
    os.makedirs(target_folder, exist_ok=True)  # 创建文件夹（如果不存在）

    if not source:  # 如果source为空，启用colab上传功能
        uploaded = files.upload()
        if uploaded:  # 确保有文件上传成功
            for fn in uploaded.keys():
                file_extension = os.path.splitext(fn)[1]  # 获取文件扩展名
                allowed_extensions = ['.jpg', '.png']  # 定义允许的文件扩展名列表
                if file_extension.lower() not in allowed_extensions:
                    print(f"文件 {fn} 格式错误，仅支持jpg或png格式，请重新上传！")
                    return
                base_name = os.path.splitext(fn)[0]
                new_fn = base_name + ".jpg"  # 统一转换为jpg格式的文件名
                count = 1
                while os.path.exists(os.path.join(target_folder, new_fn)):
                    new_fn = f"{base_name}_{count}.jpg"
                    count += 1
                image_path = os.path.join(target_folder, new_fn)
                # 先以原格式打开图片，再转换为jpg格式并保存
                with Image.open(uploaded[fn]) as im:
                    im.convert('RGB').save(image_path)
    elif source.startswith('/content/'):  # 如果是colab文件路径
        if os.path.exists(source):  # 先判断文件是否存在
            file_extension = os.path.splitext(source)[1]
            allowed_extensions = ['.jpg', '.png']
            if file_extension.lower() not in allowed_extensions:
                print(f"文件 {source} 格式错误，仅支持jpg或png格式，请检查文件！")
                return
            if file_extension.lower()!= ".jpg":  # 如果不是jpg格式，进行转换
                base_name = os.path.splitext(source)[0]
                new_source = base_name + ".jpg"
                with Image.open(source) as im:
                    im.convert('RGB').save(new_source)
                image_path = new_source
            else:
                image_path = source
        else:
            print(f" {source} 文件不存在，请检查路径是否正确！")
            return
    else:  # 否则，尝试下载网络图片
        try:
            response = requests.get(source, stream=True)
            response.raise_for_status()  # 检查请求是否成功
            file_name = os.path.basename(source)
            file_extension = os.path.splitext(file_name)[1]
            allowed_extensions = ['.jpg', '.png']
            if file_extension.lower() not in allowed_extensions:
                print(f"文件 {source} 格式错误，仅支持jpg或png格式，请更换图片链接！")
                return
            base_name = os.path.splitext(file_name)[0]
            new_fn = base_name + ".jpg"  # 统一转换为jpg格式的文件名
            image_path = os.path.join(target_folder, new_fn)
            with open(image_path, 'wb') as out_file:
                for chunk in response.iter_content(1024):
                    out_file.write(chunk)
            with Image.open(image_path) as im:
                im.convert('RGB').save(image_path)
        except requests.exceptions.RequestException as e:
            print(f"加载图片失败: {e}")
            return

    if image_path:  # 确保获取到了有效的图片路径后再显示图片
        try:        # 显示图片
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            return image_path
        except Exception as e:
            print(f"显示图片时出错: {e}")
    else:
        print("未能获取到有效的图片路径，无法显示图片。")

# -- star
download_all_models(models_info)
install_dependencies()
fix()