# -- CUDA 环境准备
import os
import sys
import glob
import subprocess

def prepare_cuda_environment():
    cuda_packages = [
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ]

    # 始终确保 CUDA 12 依赖安装完成
    for package in cuda_packages:
        print(f"正在检查/安装 CUDA 依赖: {package}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                package,
            ],
            text=True,
            capture_output=True
        )

        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"CUDA 依赖安装失败: {package}")

    nvidia_roots = (
        glob.glob("/usr/local/lib/python*/dist-packages/nvidia")
        + glob.glob("/usr/local/lib/python*/site-packages/nvidia")
    )

    if not nvidia_roots:
        raise RuntimeError("没有找到 NVIDIA CUDA 库目录")

    nvidia_root = nvidia_roots[0]

    cuda12_paths = [
        f"{nvidia_root}/cuda_nvrtc/lib",
        f"{nvidia_root}/cuda_runtime/lib",
        f"{nvidia_root}/cublas/lib",
        f"{nvidia_root}/cudnn/lib",
        f"{nvidia_root}/cufft/lib",
        f"{nvidia_root}/curand/lib",
        f"{nvidia_root}/cusolver/lib",
        f"{nvidia_root}/cusparse/lib",
        f"{nvidia_root}/nccl/lib",
        f"{nvidia_root}/nvjitlink/lib",
        "/usr/lib64-nvidia",
    ]

    cuda12_paths = [
        path for path in cuda12_paths
        if os.path.isdir(path)
    ]

    # 清理旧 LD_LIBRARY_PATH，避免 CUDA 13 路径优先或混用
    old_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")

    filtered_old_paths = [
        path for path in old_paths
        if path
        and os.path.isdir(path)
        and "/site-packages/nvidia/" not in path
        and "/dist-packages/nvidia/" not in path
    ]

    final_paths = list(dict.fromkeys(cuda12_paths + filtered_old_paths))

    os.environ["LD_LIBRARY_PATH"] = ":".join(final_paths)

    print("CUDA 12 环境准备完成")
    print("LD_LIBRARY_PATH:")
    print(os.environ["LD_LIBRARY_PATH"])

# -- 下载模型 26s
import requests
import zipfile
from concurrent.futures import ThreadPoolExecutor
from IPython.display import clear_output, display, HTML

models_info = [
    #('https://github.com/karaokenerds/python-audio-separator/releases/download/v0.12.1/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', 'onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl', '/content/roop/'),
    ('https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx', 'inswapper_128.onnx', '/content/roop/checkpoints/'),
    ('https://github.com/Hillobar/Rope/releases/download/Sapphire/inswapper_128.fp16.onnx', 'inswapper_128.fp16.onnx', '/content/roop/checkpoints/'),
    ('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', 'buffalo_l.zip', '/content/'),
    ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', 'GFPGANv1.4.pth', '/content/roop/models/'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth', 'detection_Resnet50_Final.pth', '/content/roop/gfpgan/weights/'),
    ('https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth', 'parsing_parsenet.pth', '/content/roop/gfpgan/weights/')
]

def download_model(url, name, path):
    local_path = os.path.join(path, name)
    tmp_path = local_path + '.part'

    # [修复] 已存在的文件不再重复下载；下载失败也不再静默，统一向上抛出
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"{name} 已存在，跳过下载")
    else:
        try:
            os.makedirs(path, exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            # [修复] 先写临时文件再原子替换，避免中断后残留半成品被误判为已下载
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            os.replace(tmp_path, local_path)
            print(f"{name} 下载成功!")
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"{name} 下载失败: {e}") from e

    if name == 'buffalo_l.zip':
        try:
            extract_zip(local_path, "/content/roop/checkpoints/models/buffalo_l")
            print(f"{name} 解压成功!")
        except Exception as e:
            raise RuntimeError(f"{name} 解压失败: {e}") from e

def download_all_models(models_info):
    # [修复] 收集 future 并检查异常，避免下载失败被 ThreadPool 静默吞掉
    errors = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_model, *info) for info in models_info]
        for info, future in zip(models_info, futures):
            try:
                future.result()
            except Exception as e:
                errors.append(str(e))
                print(f"❌ {info[1]}: {e}")
    if errors:
        raise RuntimeError(f"共 {len(errors)} 个模型处理失败，请查看上方日志")

def extract_zip(zip_file_path, extract_path):
    # [修复] 解压失败不再静默，向上抛出由调用方统一汇总
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        raise RuntimeError(f"解压 {zip_file_path} 失败: {e}") from e

# -- 修复degradations 3s
def fix():
    local_path = "/content/roop/degradations.py"

    if not os.path.exists(local_path):
        print(f"Local file {local_path} not found.")
        return

    candidates = glob.glob(
        "/usr/local/lib/python*/dist-packages/basicsr/data/degradations.py"
    ) + glob.glob(
        "/usr/local/lib/python*/site-packages/basicsr/data/degradations.py"
    )

    if not candidates:
        print("未找到 basicsr/data/degradations.py，跳过复制。")
        return

    try:
        subprocess.run(
            ["cp", local_path, candidates[0]],
            check=True
        )
        print(f"Copied to {candidates[0]}")
    except Exception as e:
        print(f"复制 degradations.py 失败: {e}")


# -- 安装依赖 25s
def install_dependencies():
    # 拆分较长的命令，避免一次性安装过多包导致冲突
    # [优化] 4 个 nvidia-cu12 包由 prepare_cuda_environment() 统一安装并在失败时抛错，此处不再重复
    commands = [
        'pip install --progress-bar off --quiet onnxruntime-gpu==1.20.2',
        'pip install --progress-bar off --quiet onnx',

        'pip install --progress-bar off --quiet insightface==0.7.3',
        'pip install --progress-bar off --quiet tk==0.1.0',
        'pip install --progress-bar off --quiet customtkinter==5.2.0',

        'pip install --progress-bar off --quiet --no-build-isolation git+https://github.com/Disty0/BasicSR.git@master',
        'pip install --progress-bar off --quiet --no-build-isolation --no-deps git+https://github.com/Disty0/GFPGAN.git@master',
        'pip install --progress-bar off --quiet facexlib',
        'pip install --progress-bar off --quiet "protobuf>=6.31.1"',
        'pip install --progress-bar off --quiet --no-cache-dir -I tkinterdnd2-universal==1.7.3 tkinterdnd2==0.3.0'
    ]

    for cmd in commands:
        # 执行命令并捕获 stdout 和 stderr
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True  # 以文本形式返回输出，而非字节
        )
        
        # 提取当前命令安装的包（简化处理：取最后一个参数或多个参数）
        parts = cmd.split()
        # 找到第一个非选项参数（即包名开始的位置）
        pkg_start = next(i for i, part in enumerate(parts) if not part.startswith('--'))
        packages = ' '.join(parts[pkg_start:])
        
        if result.returncode == 0:
            print(f"✅ {packages} 安装成功")
        else:
            print(f"❌ {packages} 安装失败！")
            print("错误信息：")
            print(result.stderr)

            
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
# ===== [patch] moviepy 兼容层
import subprocess as _sp
import sys as _sys
import pathlib as _pl
_sp.run([_sys.executable, '-m', 'pip', 'install', '-U', 'setuptools', 'wheel', 'pip'], check=False)
_sp.run([_sys.executable, '-m', 'pip', 'install', '-U', 'moviepy', 'imageio-ffmpeg'], check=False)
_mv = __import__('moviepy')
_pkg = _pl.Path(_mv.__file__).parent
_editor = r'''import moviepy as _m
from moviepy import *


def _pull(name):
    if name in globals():
        return
    obj = getattr(_m, name, None)
    if obj is not None:
        globals()[name] = obj
        return
    import importlib
    import pkgutil
    for _im, _mn, _ispkg in pkgutil.walk_packages(_m.__path__, _m.__name__ + '.'):
        try:
            mod = importlib.import_module(_mn)
        except Exception:
            continue
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
            return


for _n in [
    'VideoFileClip', 'AudioFileClip', 'ImageClip', 'ColorClip', 'TextClip',
    'VideoClip', 'CompositeVideoClip', 'AudioClip', 'AudioArrayClip',
    'CompositeAudioClip', 'concatenate_videoclips', 'clips_array',
    'ImageSequenceClip',
]:
    _pull(_n)
'''
(_pkg / 'editor.py').write_text(_editor)
del _sp, _sys, _pl, _mv, _pkg

import moviepy.editor as mp
from base64 import b64encode

def clean_url(url):
    parsed = urlparse(url)
    # [修复] 保留 query：Google Drive 的 ?id=&export=download、Dropbox 的 ?dl=1 依赖 query 才能下载
    if parsed.query:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parsed.query}"
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def format_size_limit_msg(size_bytes, max_file_size):
    return (
        f"文件过大！当前文件大小: {size_bytes / (1024 * 1024):.2f} MB，"
        f"最大允许大小: {max_file_size / (1024 * 1024)} MB。"
        f"建议：1. 使用网盘分享链接；2. 压缩文件；3. 选择更小的媒体文件。"
    )

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
            # [修复] HEAD 只用于提前拦截超大文件；站点不支持 HEAD(405) 时不再直接失败，降级为流式判断
            try:
                head_response = requests.head(cleaned, headers=headers, allow_redirects=True, timeout=30)
                head_response.raise_for_status()
                try:
                    content_length = int(head_response.headers.get('Content-Length', 0) or 0)
                except (TypeError, ValueError):
                    content_length = 0
                if content_length > max_file_size:
                    raise ValueError(format_size_limit_msg(content_length, max_file_size))
            except requests.exceptions.RequestException:
                pass

            response = requests.get(cleaned, headers=headers, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            # [修复] URL 以 / 结尾时 basename 为空，会导致下面的 open() 打开目录，故做兜底
            file_name = os.path.basename(urlparse(cleaned).path) or 'downloaded_media'
            media_path = Path(target_folder) / file_name
            media_path.parent.mkdir(parents=True, exist_ok=True)
            # [修复] 流式写入时实时累计，防止缺少 Content-Length 时大小限制被绕过
            downloaded = 0
            with open(media_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_file_size:
                        break
                    f.write(chunk)

            if downloaded > max_file_size:
                try:
                    os.remove(media_path)
                except OSError:
                    pass
                raise ValueError(format_size_limit_msg(downloaded, max_file_size))

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
    video = None
    try:
        video = mp.VideoFileClip(str(media_path))
        video.write_videofile(str(new_path), codec='libx264')
        return new_path
    except Exception as e:
        raise RuntimeError(f"视频格式转换失败: {e}") from e
    finally:
        # [修复] 无论成功失败都释放句柄，避免 ffmpeg 子进程残留
        if video is not None:
            try:
                video.close()
            except Exception:
                pass

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
        # [修复] 加 -y：同名残留文件会让 ffmpeg 在 stdin 等待确认而卡死
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(media_path),
            '-t', str(preview_duration),
            '-c', 'copy',
            preview_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # [修复] 降级重编码失败时同样清理预览文件，避免留下脏文件干扰下次调用
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', str(media_path),
                '-t', str(preview_duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                preview_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            if os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                except OSError:
                    pass
            raise RuntimeError(f"生成视频预览失败: {e}") from e
    try:
        with open(preview_path, 'rb') as f:
            video_data = f.read()
    finally:
        if os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except OSError:
                pass
    data_url = "data:video/mp4;base64," + b64encode(video_data).decode()
    display(HTML(f'''
    <video width=300 height=200 controls>  
        <source src="{data_url}" type="video/mp4">
    </video>
    '''))

def display_media(source, show_media=True, save_to_path=1, preview_duration=10):
    try:
        # [修复] save_to_path 分支移入 try 并补齐 else，避免 target_folder 为 None 时抛 AttributeError
        if save_to_path == 1:
            target_folder = Path("/content/source")
        elif save_to_path == 2:
            target_folder = Path("/content/target")
        else:
            raise ValueError("save_to_path 参数的值必须为 1 或 2！")
        target_folder.mkdir(parents=True, exist_ok=True)
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

# ===== [patch] 屏蔽 roop GUI，避免 tkinterdnd2/tix 在 py3.13 崩溃 =====
def patch_core():
    core = Path('/content/roop/roop/core.py')
    if not core.exists():
        print('未找到 core.py，跳过 headless 补丁')
        return
    # [修复] 按被替换行的实际缩进逐行对齐注入，避免目标 import 被缩进时报 IndentationError
    ui_block_lines = [
        '# [patch] headless',
        'class _NoUI:',
        '    def __getattr__(self, _n):',
        '        return lambda *_a, **_k: None',
        'ui = _NoUI()',
    ]
    targets = ["import roop.ui as ui", "import roop.ui", "from roop import ui"]
    lines = core.read_text().splitlines(keepends=True)
    patched = False
    for index, line in enumerate(lines):
        if line.strip() in targets:
            indent = line[:len(line) - len(line.lstrip())]
            line_ending = line[len(line.rstrip()):] or '\n'
            lines[index] = '\n'.join(indent + text for text in ui_block_lines) + line_ending
            patched = True
            break
    if not patched:
        print('未找到 roop.ui 的 import 语句，跳过替换')
        return
    core.write_text(''.join(lines))
    print('[patch] core.py 已补丁(headless)')

# -- star
prepare_cuda_environment()
install_dependencies()
fix()
patch_core()
download_all_models(models_info)
