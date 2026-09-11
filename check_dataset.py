import sys
import os

# 添加项目路径
sys.path.append(os.getcwd())

from datasets.prcc import prcc
from datasets.ltcc import ltcc


def check_dataset(dataset, name, subset='train', num_samples=50):
    """检查数据集的通用函数"""
    if subset == 'train':
        data = dataset.train
    elif subset == 'val':
        data = getattr(dataset, 'val', None)
    elif subset == 'query':
        # PRCC: the clothes-changing protocol is the canonical query split
        data = getattr(dataset, 'query', None) or dataset.query_diff
    elif subset == 'query_same':
        data = dataset.query_same
    elif subset == 'query_diff':
        data = dataset.query_diff
    elif subset == 'gallery':
        data = dataset.gallery
    else:
        raise ValueError(f"Invalid subset: {subset}")

    if data is None:
        print(f"\nDataset: {name}, subset '{subset}' is not available")
        return

    print(f"\n{'='*80}")
    print(f"Dataset: {name}, Subset: {subset.upper()}")
    print(f"{'='*80}")
    print(f"{'No.':<5} {'Image Path':<60} {'PID':<8} {'CID':<5} {'Cloth ID'}")
    print(f"{'-'*80}")

    for i, (img_path, pid, cid, cloth_id) in enumerate(data[:num_samples]):
        print(f"{i+1:<5} {img_path:<60} {pid:<8} {cid:<5} {cloth_id}")

    print(f"\nTotal in subset: {len(data)}")
    print(f"Displayed: {min(num_samples, len(data))}")


def main():
    # 配置数据集路径
    data_root = '../data'

    print("请选择要检查的数据集:")
    print("1. PRCC")
    print("2. LTCC")

    choice = input("请输入选项 (1/2): ").strip()

    print("\n请选择要检查的子集:")
    print("1. train")
    print("2. query")
    print("3. gallery")
    print("4. query_same (PRCC test/B)")
    print("5. query_diff (PRCC test/C)")
    print("6. val")

    subset_choice = input("请输入选项 (1-6): ").strip()
    subset_map = {'1': 'train', '2': 'query', '3': 'gallery',
                  '4': 'query_same', '5': 'query_diff', '6': 'val'}
    subset = subset_map.get(subset_choice)
    if subset is None:
        print("无效子集")
        return

    num_samples = 50  # 可自定义

    if choice == '1':
        dataset = prcc(root=data_root, verbose=False)
        check_dataset(dataset, 'PRCC', subset, num_samples)
    elif choice == '2':
        dataset = ltcc(root=data_root, verbose=False)
        check_dataset(dataset, 'LTCC', subset, num_samples)
    else:
        print("无效选择")


if __name__ == '__main__':
    main()
