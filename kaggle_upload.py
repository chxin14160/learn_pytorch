# kaggle_upload.py - 准备上传到Kaggle的文件
import os
import shutil
from pathlib import Path
import zipfile


def prepare_kaggle_upload():
    """准备上传到Kaggle的文件包"""

    # 创建上传目录
    upload_dir = Path("kaggle_upload")
    # upload_dir = Path("F:\PycharmProjects\learn_pytorch")
    upload_dir.mkdir(exist_ok=True)

    # 需要上传的文件
    files_to_upload = [
        "common.py",
        "learn_ComputerVision.py",
        # "README.md",
        "requirements.txt"
    ]

    # 复制文件到上传目录
    for file_path in files_to_upload:
        if Path(file_path).exists():
            shutil.copy2(file_path, upload_dir / file_path)
            print(f"📁 已复制: {file_path}")

    # 创建Kaggle专用的requirements.txt
    requirements = """torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
matplotlib>=3.4.0
Pillow>=8.3.0
"""
    requirements = """torch>=1.10.2
torchvision>=0.11.3
numpy>=1.24.0
matplotlib>=3.7.5
pandas>=2.3.3
Pillow>=10.2.0
sympy==1.13.3
tensorflow==2.20.0
"""

    with open(upload_dir / "requirements.txt", "w") as f:
        f.write(requirements)

    # 创建说明文件
    readme_content = """# PyCharm + Kaggle GPU训练项目
                        
## 使用方法
1. 在Kaggle中上传此文件夹
2. 创建新的Notebook
3. 复制kaggle_train.py的内容到Notebook
4. 开启GPU加速器
5. 运行代码

## GPU设置步骤
1. 在Notebook页面，点击右侧的 **Settings**
2. 找到 **Accelerator** 选项
3. 选择 **GPU**
4. 开启 **Internet**（如果需要下载数据）

## 文件说明
- kaggle_train.py: 主要训练脚本
- requirements.txt: 依赖包列表

## 注意事项
- 确保开启GPU加速
- 数据集会自动下载
- 训练完成后下载模型文件
"""

    with open(upload_dir / "README.md", "w") as f:
        f.write(readme_content)

    # 创建ZIP包（可选）
    zip_path = "kaggle_upload.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in upload_dir.rglob("*"):
            zipf.write(file_path, file_path.relative_to(upload_dir))

    print(f"✅ 上传文件准备完成！")
    print(f"📁 上传目录: {upload_dir}")
    print(f"📦 ZIP文件: {zip_path}")
    print("📦 请上传 'kaggle_upload' 文件夹到Kaggle")
    print("💡 或者直接上传 'kaggle_upload.zip' 文件")


def create_requirements():
    """创建requirements.txt"""
    requirements = """torch==1.10.0
torchvision==0.11.0
numpy
pandas
matplotlib
kaggle
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)

    print("✅ requirements.txt 已创建")


if __name__ == "__main__":
    # create_requirements()
    prepare_kaggle_upload()



