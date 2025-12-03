# kaggle_setup.py 创建配置文件，配置Kaggle API
import os
import json
from pathlib import Path


def create_kaggle_json_manually():
    """手动创建kaggle.json文件"""
    # Kaggle账号信息（需要手动填写）
    kaggle_config = {
        "username": "-chxin14160-",  # 例如: "johnsmith"
        "key": "—KGAT_fc4bf87eca6bcf55f301bf66a1295f—"  # 例如: "a1b2c3d4e5f6g7h8i9j0"
    }

    # 确定保存路径
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)  # 创建目录

    kaggle_json_path = kaggle_dir / 'kaggle.json'

    # 写入文件
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_config, f, indent=4)

    # 设置文件权限（Linux/Mac需要）
    if os.name != 'nt':  # 非Windows系统
        os.chmod(kaggle_json_path, 0o600)

    print(f"✅ kaggle.json 已创建在: {kaggle_json_path}")
    return kaggle_json_path


def verify_kaggle_setup():
    """验证Kaggle配置"""
    kaggle_dir = Path.home() / '.kaggle'
    config_file = kaggle_dir / 'kaggle.json'

    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

        print("🎉 Kaggle配置验证成功!")
        print(f"   用户名: {config.get('username')}")
        print(f"   API Key: {config.get('key')[:10]}***")
        return True
    else:
        print("❌ Kaggle配置失败，请重新运行setup_kaggle.py")
        return False


# if __name__ == "__main__":
# create_kaggle_json_manually()
verify_kaggle_setup()


