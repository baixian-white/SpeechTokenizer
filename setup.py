# setup.py
"""
脚本名称: setup.py
功能描述: 
    这是 Python 项目的标准打包与分发配置文件。
    它的核心作用是告诉 pip 工具如何安装这个包，包括：
    1. 包的元数据（名称、版本、作者）。
    2. 需要包含哪些代码文件夹（packages）。
    3. 运行该包需要依赖哪些第三方库（install_requires）。

使用场景:
    - 本地安装: pip install -e . (编辑模式) 或 pip install .
    - 打包发布: python setup.py sdist bdist_wheel
"""

from pathlib import Path
from setuptools import setup

# ==========================================
# 1. 项目元数据配置
# ==========================================
NAME = 'speechtokenizer'                            # 包名 (pip install speechtokenizer)
DESCRIPTION = 'Unified speech tokenizer for speech language model' # 简短描述
URL = 'https://github.com/ZhangXInFD/SpeechTokenizer' # 项目主页/Github地址
EMAIL = 'xin_zhang22@m.fudan.edu.cn'                # 作者邮箱
AUTHOR = 'Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, Xipeng Qiu' # 作者列表
REQUIRES_PYTHON = '>=3.8.0'                         # Python 版本要求

# ==========================================
# 2. 动态获取版本号
# ==========================================
# 为了避免在 setup.py 和 __init__.py 中写两遍版本号（容易导致不一致），
# 这里直接读取 __init__.py 中的 __version__ 变量。
for line in open('speechtokenizer/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']
        
HERE = Path(__file__).parent

# ==========================================
# 3. 读取长描述 (README.md)
# ==========================================
# 尝试读取 README.md 作为 PyPI 主页的详细介绍。
try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
    
# ==========================================
# 4. 核心 Setup 函数
# ==========================================
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown', # 指定 README 是 Markdown 格式
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,

    # 🌟 关键点：指定包含的包路径
    # 这里显式列出了所有包含 __init__.py 的子目录。
    # 如果您在开发 NAS 时新建了文件夹（如 speechtokenizer.nas），
    # 必须把 'speechtokenizer.nas' 加到这里，否则安装时会报错找不到模块！
    packages=[
        'speechtokenizer', 
        'speechtokenizer.quantization', 
        'speechtokenizer.modules', 
        'speechtokenizer.trainer'
    ],

    # 🌟 依赖管理
    # 用户 pip install 本项目时，pip 会自动检查并下载这些库
    install_requires=[
        'numpy', 
        'torch', 
        'torchaudio', 
        'einops',
        'scipy',
        'huggingface-hub',
        'soundfile', 
        'matplotlib', 
        'lion_pytorch', 
        'accelerate' # 分布式训练库，NAS 训练脚本中用到了
    ],

    include_package_data=True, # 配合 MANIFEST.in 使用，用于打包非代码文件（如配置、数据等）
    license='Apache License 2.0',
    
    # PyPI 分类标签，用于搜索过滤
    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
    ]
)