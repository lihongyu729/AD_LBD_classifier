import os
import json
import torch
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityd,
    SaveImaged
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置 =================
# 原始 JSON 路径

json_path = "/webdav/MyData/MRI/data/full_dataset.json"  
# 原始数据根目录
data_root = "/webdav/MyData/MRI/data/classifier" 
# 新的保存目录 (预处理后的数据放这里)
output_root = "/webdav/MyData/MRI/data/MRI_Preprocessed_1mm/" 
# 新的 JSON 保存路径
new_json_path = "/webdav/MyData/MRI/data/3dino_cls.fixed_1mm.json"
# =======================================

def preprocess():
    """
    函数作用：
    - 批量将 MRI 影像统一方向为 RAS，并在物理空间重采样至 1.0mm 各向同性。
    - 预处理结果保存为 .nii 文件，且严格按原始分类目录（AD、LBD、MCI）分别存放到 output_root 的子目录。
    - 生成新的 JSON，路径指向分类后的输出文件，并记录 spacing 为 [1.0, 1.0, 1.0]。
    
    设计原因：
    - 通过显式的类别解析函数将输出组织与输入分类保持一致，避免混淆。
    - 保存阶段使用手动保存以支持按样本动态选择输出目录（Compose 内的固定 output_dir 不满足该需求）。
    - 保留对损坏文件的跳过策略，确保长时间批处理不会因个别坏样本中断。
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    with open(json_path, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)

    # 统一展开 JSON 为“样本字典列表”，避免遍历到字符串键导致 item.copy() 报错
    process_list = []
    if isinstance(data_loaded, list):
        for item in data_loaded:
            if not isinstance(item, dict):
                continue
            new_item = item.copy()
            img_path = item.get('image', '')
            # 若已是绝对路径，直接使用；否则拼接 data_root
            if os.path.isabs(img_path):
                new_item['image'] = img_path
            else:
                new_item['image'] = os.path.join(data_root, img_path)
            process_list.append(new_item)
    elif isinstance(data_loaded, dict):
        for split_key, lst in data_loaded.items():
            if not isinstance(lst, list):
                continue
            for item in lst:
                if not isinstance(item, dict):
                    continue
                new_item = item.copy()
                img_path = item.get('image', '')
                if os.path.isabs(img_path):
                    new_item['image'] = img_path
                else:
                    new_item['image'] = os.path.join(data_root, img_path)
                process_list.append(new_item)
    else:
        raise ValueError("JSON 结构既不是列表也不是字典，无法处理")

    # 新增：预过滤损坏或不完整的 .nii 文件（不读取全数据，仅用头信息与文件大小判断）
    def is_valid_nii(nii_path: str) -> bool:
        """
        函数作用：
        - 校验 .nii 文件是否完整：用 NIfTI 头的数据形状与数据类型计算期望字节数，
          与实际文件大小减去数据起始偏移对比；若不足则判定为损坏。
        设计原因：
        - nibabel 在加载 ArrayProxy 时会读取整块数据，损坏文件会导致 OSError；
          通过轻量级的头部检查提前剔除问题文件，避免在 DataLoader/变换阶段抛错。
        """
        try:
            import nibabel as nib
            import numpy as np
            if not os.path.exists(nii_path):
                return False
            img = nib.load(nii_path)  # 不触发全量读取
            hdr = img.header
            shape = hdr.get_data_shape()
            dtype = np.dtype(hdr.get_data_dtype())
            expected_bytes = int(np.prod(shape) * dtype.itemsize)
            data_offset = int(hdr.get_data_offset())
            actual_bytes = os.path.getsize(nii_path) - data_offset
            return actual_bytes >= expected_bytes
        except Exception:
            return False

    valid_list = []
    skipped_files = []
    for entry in process_list:
        p = entry.get("image", "")
        if is_valid_nii(p):
            valid_list.append(entry)
        else:
            skipped_files.append(p)
    print(f"预过滤完成：有效 {len(valid_list)}，跳过损坏/不可读 {len(skipped_files)}")

    # 用 Compose 包裹变换流水线；输出扩展名改为 .nii
    from monai.transforms import Compose
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # 统一方向 (RAS 是标准医学方向)
        Orientationd(keys=["image"], axcodes="RAS"),
        # 重采样到 1.0 x 1.0 x 1.0（bilinear 更适合 MRI 的连续强度）
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear")
    ])

    # 顺序执行变换并在主线程捕获异常，遇到问题样本跳过不终止流程
    # 新增：解析类别目录（严格映射到 AD/LBD/MCI；大小写不敏感）
    def get_class_dir(path_str: str) -> str:
        """
        函数作用：
        - 根据原始影像路径推断其类别目录，返回固定的 'AD'、'LBD' 或 'MCI'（若无法匹配则返回 'UNKNOWN'）。
        
        设计原因：
        - 你的分类目录是固定集合：/webdav/MyData/MRI/data/classifier/AD、/LBD、/MCI。
          优先以 data_root 为参考取其下首级目录；若路径不在 data_root 之下，则在路径各级中匹配类别关键词。
        """
        class_map = {"ad": "AD", "lbd": "LBD", "mci": "MCI"}
        # 统一路径分隔符
        norm_path = path_str.replace("\\", "/")
        base_root = data_root.replace("\\", "/")
        # 优先使用相对 data_root 的首级目录
        try:
            rel = os.path.relpath(norm_path, base_root).replace("\\", "/")
            parts = [p for p in rel.split("/") if p and p not in (".", "..")]
            if parts:
                key = parts[0].lower()
                if key in class_map:
                    return class_map[key]
        except Exception:
            pass
        # 回退：在全路径各级中匹配类别关键词
        for seg in norm_path.split("/"):
            k = seg.lower()
            if k in class_map:
                return class_map[k]
        return "UNKNOWN"

    new_data_list = []
    import re
    import nibabel as nib
    import numpy as np
    from tqdm import tqdm
    for i, sample in tqdm(enumerate(valid_list), total=len(valid_list)):
        try:
            out = transforms(sample)

            # 解析类别并构建输出子目录（严格为 AD/LBD/MCI）
            class_dir = get_class_dir(sample["image"])
            out_dir = os.path.join(output_root, class_dir)
            os.makedirs(out_dir, exist_ok=True)

            # 构造新文件名（输出为 .nii）
            original_name = os.path.basename(sample["image"])
            file_name_no_ext = re.sub(r'(\.nii(\.gz)?)$', '', original_name, flags=re.I)
            new_filename = f"{file_name_no_ext}_1mm.nii"
            save_path = os.path.join(out_dir, new_filename)

            # 从变换结果中取数据与仿射，去掉通道维后保存为 .nii
            img_np = out["image"]
            if img_np.ndim == 4 and img_np.shape[0] == 1:
                img_np = img_np[0]
            img_np = np.asarray(img_np)
            affine = out.get("image_meta_dict", {}).get("affine", np.eye(4))
            nib.save(nib.Nifti1Image(img_np, affine), save_path)

            new_entry = {
                "image": save_path,
                "label": sample.get("label"),
                "spacing": [1.0, 1.0, 1.0],
            }
            new_data_list.append(new_entry)
        except Exception as e:
            print(f"跳过样本：{sample.get('image')}，原因：{e}")
            continue

    # 保存新的 JSON
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data_list, f, ensure_ascii=False, indent=4)
    print(f"预处理完成！新 JSON 已保存至: {new_json_path}；有效样本 {len(new_data_list)}，总跳过 {len(skipped_files)}")

if __name__ == "__main__":
    from monai.utils import set_determinism
    set_determinism(0)
    preprocess()