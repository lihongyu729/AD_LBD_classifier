import os
import json
from tqdm import tqdm

# ================= 配置 =================
# 1. 原始数据根目录：请确保这个路径下直接包含 AD, LBD, MCI 等分类文件夹
#    根据之前的对话，您的数据可能在 'classifier' 或 'classifier_splits' 下
#    请根据实际情况选择或修改。
#    示例: "/webdav/MyData/MRI/data/classifier"
#    示例: "/webdav/MyData/MRI/data/classifier_splits/train"
DATA_ROOT = "/webdav/MyData/MRI/data/classifier"

# 2. 输出 JSON 文件的路径：这是新生成的完整数据集清单
OUTPUT_JSON_PATH = "/webdav/MyData/MRI/data/full_dataset.json"

# 3. 分类名称与标签的映射关系：文件夹名 -> 数字标签
#    请确保这里的键（'AD', 'LBD', 'MCI'）与您 DATA_ROOT 下的文件夹名完全一致（大小写敏感）
CLASS_TO_LABEL = {
    "AD": 0,
    "LBD": 1,
    "MCI": 2
}
# =======================================

def create_full_dataset_json():
    """
    函数作用：
    - 扫描指定的数据根目录（DATA_ROOT），查找所有分类子目录（如 AD, LBD, MCI）。
    - 遍历每个分类子目录，搜集所有 .nii 和 .nii.gz 文件的绝对路径。
    - 根据文件所在的目录和 CLASS_TO_LABEL 映射，为每个文件分配一个数字标签。
    - 将所有文件的信息（图像路径和标签）汇集，生成一个完整的 JSON 数据集清单。

    设计原因：
    - 解决原始 JSON 文件不完整，导致预处理脚本（process.py）只处理了部分数据的问题。
    - 通过自动扫描文件系统来创建数据清单，确保所有符合条件的数据都被包含进来。
    - 生成的 JSON 格式（一个包含字典的列表）能直接被 MONAI 的 Dataset 或后续的 process.py 脚本使用。
    """
    if not os.path.isdir(DATA_ROOT):
        print(f"错误：数据根目录 '{DATA_ROOT}' 不存在或不是一个目录。请检查路径配置。")
        return

    dataset_list = []
    print(f"开始扫描目录: {DATA_ROOT}")
    print(f"查找的分类文件夹: {list(CLASS_TO_LABEL.keys())}")

    # 使用 os.walk 遍历，可以处理更深层级的目录结构
    for root, dirs, files in tqdm(os.walk(DATA_ROOT), desc="扫描文件夹"):
        for file_name in files:
            if file_name.endswith((".nii", ".nii.gz")):
                # 从路径中解析出所属的类别
                # 逻辑：从当前文件路径向上追溯，找到第一个匹配 CLASS_TO_LABEL 的目录名
                current_class = None
                path_parts = root.replace("\\", "/").split("/")
                for part in reversed(path_parts):
                    if part in CLASS_TO_LABEL:
                        current_class = part
                        break
                
                if current_class:
                    file_path = os.path.join(root, file_name)
                    label = CLASS_TO_LABEL[current_class]
                    
                    dataset_list.append({
                        "image": file_path.replace("\\", "/"), # 统一使用斜杠
                        "label": label
                    })
                else:
                    # 如果一个 .nii 文件不在任何一个指定的分类目录下，则打印警告
                    tqdm.write(f"警告：跳过文件 '{os.path.join(root, file_name)}'，因为它不属于任何一个已定义的分类目录。")


    if not dataset_list:
        print("警告：没有在指定目录下找到任何 .nii 或 .nii.gz 文件。请确认 DATA_ROOT 和 CLASS_TO_LABEL 配置是否正确。")
        return

    print(f"\n扫描完成！总共找到 {len(dataset_list)} 个影像文件。")

    # 保存为 JSON 文件
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset_list, f, ensure_ascii=False, indent=4)
        print(f"成功生成完整的数据集清单: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"错误：保存 JSON 文件失败。原因: {e}")

if __name__ == "__main__":
    create_full_dataset_json()