import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import bisect
from .basic_loader import compute_train_statistics

class UottawaDataset(Dataset):
    def __init__(self, file_paths, win_len, stride, mode='all', few_shot_num=3, 
                 class_map=None, transform_params=None):
        self.file_paths = file_paths
        self.win_len = win_len
        self.stride = stride
        self.mode = mode
        self.few_shot_num = few_shot_num
        self.class_map = class_map if class_map is not None else {}
        self.unknown_id = len(self.class_map)
        self.trans = transform_params if transform_params else {}

        raw_mean = self.trans.get('vib_mean', 0.0)
        self.vib_mean = np.array(raw_mean).flatten()[0]
        raw_std = self.trans.get('vib_std', 1.0)
        self.vib_std = np.array(raw_std).flatten()[0]
        self.rpm_min, self.rpm_max = self.trans.get('rpm_limits', (600.0, 1800.0))
        self.rpm_span = self.rpm_max - self.rpm_min if self.rpm_max > self.rpm_min else 1.0
        self.indices = [] 
        self.total_samples = 0
        self.cache = {}
        self.cumulative_indices = []
        
        self._build_index_map()

    def _build_index_map(self):
        current_cumulative = 0
        self.indices = []
        
        for f_path in self.file_paths:
            try:
                label_id = self._parse_label(f_path)
                shape = np.load(f_path, mmap_mode='r').shape
                n_points = shape[0]
                
                if n_points < self.win_len:
                    n_windows = 0
                else:
                    n_windows = (n_points - self.win_len) // self.stride + 1
                
                if self.mode == 'few_shot':
                    n_windows = min(n_windows, self.few_shot_num)
                
                if n_windows > 0:
                    self.indices.append((f_path, label_id, current_cumulative, n_windows))
                    current_cumulative += n_windows
                    
            except Exception as e:
                print(f"Warning: Error reading metadata for {f_path}: {e}")
        
        self.total_samples = current_cumulative
        self.cumulative_indices = [x[2] + x[3] for x in self.indices]

    def _parse_label(self, f_path):
        path_norm = f_path.replace('\\', '/').lower()
        for cls_name, idx in self.class_map.items():
            if cls_name.lower() in path_norm:
                return idx
        return self.unknown_id

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_indices, idx)
        f_path, label_id, group_start_cum, _ = self.indices[file_idx]
        
        local_idx = idx - group_start_cum
        start_pt = local_idx * self.stride
        
        if f_path not in self.cache:
            self.cache[f_path] = np.load(f_path).astype(np.float32)
        full_data = self.cache[f_path]
        
        end_pt = start_pt + self.win_len
        segment = full_data[start_pt:end_pt, :]
        
        return self._process_segment(segment, label_id)

    def _process_segment(self, segment, label_id):
        raw_vib = segment[:, 0]
        raw_rpm = segment[:, 1]
        norm_vib = (raw_vib - self.vib_mean) / (self.vib_std + 1e-6)

        norm_rpm = 2.0 * (raw_rpm - self.rpm_min) / self.rpm_span - 1.0

        x_tensor = torch.from_numpy(norm_vib[np.newaxis, :]).float()
        c_tensor = torch.from_numpy(norm_rpm[np.newaxis, :]).float()
        y_tensor = torch.tensor(label_id, dtype=torch.long)
        
        return x_tensor, c_tensor, y_tensor

def get_uottawa_loaders(data_root, batch_size, win_len, stride, few_shot_num, 
                        training_classes, testing_classes):
    class_map = {}
    for i, cls in enumerate(training_classes):
        class_map[cls] = i
    unknown_classes = [c for c in testing_classes if c not in training_classes]
    for i, cls in enumerate(unknown_classes):
        class_map[cls] = len(training_classes) + i
    all_files = sorted(glob.glob(os.path.join(data_root, "**", "*.npy"), recursive=True))
    
    def filter_files(file_list, target_classes):
        selected = []
        for f in file_list:
            f_norm = f.replace('\\', '/').lower() 
            for cls_name in target_classes:
                if cls_name.lower() in f_norm:
                    selected.append(f)
                    break
        return selected
    
    train_files = []
    val_files = []
    known_test_part = []
    for cls in training_classes:
        cls_files = filter_files(all_files, [cls])
        n_files = len(cls_files)
        if n_files == 0:
            print(f"[Warning] Class '{cls}' has no files! Skipping...")
            continue
        if n_files < 2:
            print(f"[Warning] Class '{cls}' only has {n_files} file. Putting into Train set.")
            train_files.extend(cls_files)
            continue
        cls_train, cls_temp = train_test_split(cls_files, test_size=0.3, random_state=42, shuffle=True)
        if len(cls_temp) > 0:
            if len(cls_temp) == 1:
                 known_test_part.extend(cls_temp)
            else:
                cls_val, cls_test = train_test_split(cls_temp, test_size=2/3, random_state=42, shuffle=True)
                val_files.extend(cls_val)
                known_test_part.extend(cls_test)
        else:
            pass

        train_files.extend(cls_train)
    unknown_classes = [c for c in testing_classes if c not in training_classes]
    unknown_test_part = []
    
    if unknown_classes:
        unknown_files_all = filter_files(all_files, unknown_classes)
        if unknown_files_all:
            unknown_test_part, _ = train_test_split(unknown_files_all, test_size=0.8, random_state=42, shuffle=True)
            
    final_test_files = known_test_part + unknown_test_part
    
    print(f"\n[Split Info (Stratified)]")
    print(f"  Train: {len(train_files)} files | Val  : {len(val_files)} files")
    print(f"  Test : {len(final_test_files)} files (Known: {len(known_test_part)}, Unknown: {len(unknown_test_part)})")

    vib_mean, vib_std = compute_train_statistics(train_files, channel_indices=[0])
    rpm_limits = (600.0, 1800.0)
    print(f"[Config] Using Fixed Limits for RPM: {rpm_limits}")

    trans_params = {
        'vib_mean': vib_mean,
        'vib_std': vib_std,
        'rpm_limits': rpm_limits
    }
    
    labeled_ds = UottawaDataset(train_files, win_len, stride, 
                                mode='few_shot', few_shot_num=few_shot_num,
                                class_map=class_map, transform_params=trans_params)
    
    unlabeled_ds = UottawaDataset(train_files, win_len, stride, 
                                  mode='all',
                                  class_map=class_map, transform_params=trans_params)
    
    val_ds = UottawaDataset(val_files, win_len, stride, 
                            mode='all',
                            class_map=class_map, transform_params=trans_params)
    
    test_ds = UottawaDataset(final_test_files, win_len, stride, 
                             mode='all',
                             class_map=class_map, transform_params=trans_params)

    kwargs = {'num_workers': 4, 'pin_memory': True} 
    
    return (
        DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs),
        DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    )