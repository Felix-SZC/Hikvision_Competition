import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class MVTecDataset(Dataset):
    """MVTec风格的数据集加载器"""
    
    def __init__(self, root_path, category_id, split='train', transform=None, defect_samples_for_threshold=5):
        """
        Args:
            root_path (str): 数据集根目录，例如 'dataset'
            category_id (int or str): 类别 ID，例如 1
            split (str): 'train', 'threshold', 'test'
                - 'train': 加载 normal 样本用于构建 Memory Bank
                - 'threshold': 加载部分 NG 样本用于阈值搜索 (满足比赛 Ni >= 5)
                - 'test': 加载剩余 NG 样本用于评估 (如果有)
            transform (callable, optional): transform to be applied on a sample.
            defect_samples_for_threshold (int): 用于阈值搜索的缺陷样本数量
        """
        self.root_path = root_path
        self.category_id = str(category_id)
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.gt_paths = []
        self.labels = []  # 0: normal, 1: anomaly

        self._load_data(defect_samples_for_threshold)

    def _load_data(self, defect_samples_for_threshold):
        ok_dir = os.path.join(self.root_path, self.category_id, f"{self.category_id}-ok")
        ng_dir = os.path.join(self.root_path, self.category_id, f"{self.category_id}_ng")
        mark_file = os.path.join(ng_dir, "mark.txt")

        # Load Normal Samples (Training data for PatchCore)
        if self.split == 'train':
            if os.path.exists(ok_dir):
                normal_images = glob.glob(os.path.join(ok_dir, "*.bmp"))
                self.image_paths.extend(normal_images)
                self.labels.extend([0] * len(normal_images))
                self.gt_paths.extend([None] * len(normal_images))
        
        # Load Defect Samples
        if self.split in ['threshold', 'test'] and os.path.exists(mark_file):
            defect_data = []
            try:
                with open(mark_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_name = parts[0]
                            mask_name = parts[1]
                            defect_data.append((
                                os.path.join(ng_dir, img_name),
                                os.path.join(ng_dir, mask_name)
                            ))
            except Exception as e:
                print(f"读取标记文件 {mark_file} 时出错: {e}")
            
            # Split defect samples
            if len(defect_data) > 0:
                if self.split == 'threshold':
                    selected_defects = defect_data[:defect_samples_for_threshold]
                else: # test
                    selected_defects = defect_data[defect_samples_for_threshold:]
                    
                for img_path, mask_path in selected_defects:
                    self.image_paths.append(img_path)
                    self.gt_paths.append(mask_path)
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        mask = torch.zeros((1, 224, 224)) # Default empty mask
        if self.gt_paths[idx] is not None:
            mask = Image.open(self.gt_paths[idx]).convert('L')
            mask = mask.resize((224, 224), Image.NEAREST)
            mask = transforms.ToTensor()(mask)
        
        return image, label, mask, img_path

