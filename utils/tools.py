import numpy as np
import torch
from scipy.stats import norm, gamma
import os
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def get_adaptive_threshold(scores, method='3sigma'):
        if method == '3sigma':
            mu = np.mean(scores)
            std = np.std(scores)
            threshold = mu + 3 * std
            return threshold, f"Mean + 3*Std ({mu:.4f} + 3*{std:.4f})"
            
        elif method == 'iqr':
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
            return threshold, f"Q3 + 1.5*IQR ({q3:.4f} + 1.5*{iqr:.4f})"
            
        elif method == 'percentile95':
            threshold = np.percentile(scores, 95)
            return threshold, "95th Percentile"
            
        elif method == 'gamma':
            try:
                shape, loc, scale = gamma.fit(scores)
                threshold = gamma.ppf(0.99, shape, loc=loc, scale=scale)
                return threshold, "Gamma Distribution (99% Conf)"
            except:
                mu = np.mean(scores)
                std = np.std(scores)
                return mu + 3 * std, "Mean + 3*Std (Fallback)"
        
        else:
            raise ValueError("Unknown threshold method")

def plot_tsne(args, all_features, all_labels, known_mask, unknown_mask, all_rpms, epoch=None):
    print("\nComputing t-SNE... (this may take a while)")
    MAX_SAMPLES = 30000

    indices_known = np.where(known_mask)[0]
    indices_unknown = np.where(unknown_mask)[0]

    if len(indices_known) > MAX_SAMPLES:
        indices_known_sampled = np.random.choice(indices_known, size=MAX_SAMPLES, replace=False)
    else:
        indices_known_sampled = indices_known

    indices_sampled = np.concatenate([indices_known_sampled, indices_unknown])

    features_tsne = all_features[indices_sampled]
    labels_tsne = all_labels[indices_sampled]
    mask_tsne = known_mask[indices_sampled]
    rpms_tsne = all_rpms[indices_sampled]
    
    print(f"t-SNE Sampling: Original {len(all_features)} -> Sampled {len(features_tsne)} "
        f"(Includes all {len(indices_unknown)} unknown samples)")

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    z_2d = tsne.fit_transform(features_tsne)

    plot_data = pd.DataFrame({
        'x': z_2d[:, 0],
        'y': z_2d[:, 1],
        'label': labels_tsne,
        'is_known': mask_tsne,
        'rpm': rpms_tsne
    })

    ordered_classes = args.training_classes + [c for c in args.testing_classes if c not in args.training_classes]

    plot_data['class_name'] = plot_data['label'].apply(
        lambda x: ordered_classes[x] if x < len(ordered_classes) else f'Outlier_{x}'
    )

    known_names = args.training_classes
    all_names = args.testing_classes
    all_unknown_names = [name for name in all_names if name not in known_names]
    
    import matplotlib.lines as mlines
    import matplotlib.colors as mcolors

    tab10 = sns.color_palette("tab10", 10)
    tab10_reversed = tab10[::-1]
    
    color_map = {}
    for i, name in enumerate(known_names):
        color_map[name] = tab10[i % 10]
    for i, name in enumerate(all_unknown_names):
        color_map[name] = tab10_reversed[i % 10]

    known_markers = ['o', 's', 'D', 'v', '^', '<', '>']
    unknown_markers = ['X', 'P', '*', 'h', 'H', '+', 'd']
    
    if len(known_names) > len(known_markers):
        known_markers = known_markers * (len(known_names) // len(known_markers) + 1)
    if len(all_unknown_names) > len(unknown_markers):
        unknown_markers = unknown_markers * (len(all_unknown_names) // len(unknown_markers) + 1)
    
    marker_map = {}
    for i, name in enumerate(known_names): 
        marker_map[name] = known_markers[i]
    for i, name in enumerate(all_unknown_names): 
        marker_map[name] = unknown_markers[i]

    plt.figure(figsize=(5, 4.5))

    rpm_min = plot_data['rpm'].min()
    rpm_max = plot_data['rpm'].max()

    for name in all_names:
        subset = plot_data[plot_data['class_name'] == name]
        if len(subset) == 0:
            continue
        
        is_known = subset['is_known'].iloc[0]
        base_color = color_map[name]

        rpm_norm = (subset['rpm'] - rpm_min) / (rpm_max - rpm_min + 1e-8)
        alphas = 0.05 + 0.95 * rpm_norm 

        rgba_colors = np.zeros((len(subset), 4))
        rgba_colors[:, :3] = mcolors.to_rgb(base_color)
        rgba_colors[:, 3] = alphas 

        size = 12
        edge_color = 'white' if is_known else 'black'
        line_width = 0.2
        
        plt.scatter(
            subset['x'], subset['y'],
            c=rgba_colors,
            marker=marker_map[name],
            s=size,
            edgecolors=edge_color,
            linewidths=line_width,
            label='_nolegend_'
        )
    
    custom_handles = []
    custom_labels = []

    for name in known_names:
        h = mlines.Line2D([], [], color=color_map[name], marker=marker_map[name], linestyle='None', markersize=8)
        custom_handles.append(h)
        custom_labels.append(name)

    if len(all_unknown_names) > 0:
        custom_handles.append(mlines.Line2D([], [], color='none'))
        custom_labels.append("--- Unknowns ---")
        for name in all_unknown_names:
            h = mlines.Line2D([], [], color=color_map[name], marker=marker_map[name], linestyle='None', markersize=10, markeredgecolor='white')
            custom_handles.append(h)
            custom_labels.append(f"{name} (Unknown)")

    custom_handles.append(mlines.Line2D([], [], color='none'))
    custom_labels.append("--- Speed (RPM) ---")

    custom_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, alpha=1.0))
    custom_labels.append("High Speed (Dark)")
    custom_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, alpha=0.3))
    custom_labels.append("Low Speed (Light)")
        
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(handles=custom_handles, labels=custom_labels, 
               loc='upper right', fontsize=9, framealpha=0.5, 
               handletextpad=0.1, labelspacing=0.3)

    plt.tight_layout(pad=0.5)
    
    if epoch is not None:
        plt.savefig(os.path.join(args.log_path, f'tsne_visualization_{epoch+1}.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(os.path.join(args.log_path, f'tsne_visualization.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_nll_distribution(args, dist_known, dist_unknown, threshold, epoch=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))

    dist_known_safe = np.array(dist_known) + 1e-6
    dist_unknown_safe = np.array(dist_unknown) + 1e-6
    threshold_safe = threshold + 1e-6

    sns.histplot(dist_known_safe, color="#1f77b4", label="Known Classes", 
                    kde=True, stat="density", bins=50, alpha=0.5, edgecolor="white", log_scale=True)
    
    sns.histplot(dist_unknown_safe, color="#d62728", label="Unknown (OOD) Classes", 
                    kde=True, stat="density", bins=50, alpha=0.5, edgecolor="white", log_scale=True)

    plt.axvline(x=threshold_safe, color='black', linestyle='--', linewidth=2.5, 
                label=f'Rejection Threshold\n(Score = {threshold:.1f})')

    plt.xlabel('Latent Distance Score (Log Scale)', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=13, loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    filename = f'nll_distribution_{epoch+1}.png' if epoch is not None else 'nll_distribution.png'
    plt.savefig(os.path.join(args.log_path, filename), dpi=300)
    plt.close()

    sns.reset_defaults()

def evaluate_open_set_and_plot_cm(args, all_scores, all_labels, all_preds, known_mask, unknown_mask, threshold, best_acc):
    unknown_cls_id = args.num_known_classes
    open_set_preds = all_preds.copy()
    open_set_preds[all_scores > threshold] = unknown_cls_id
    
    open_set_labels = all_labels.copy()
    open_set_labels[unknown_mask] = unknown_cls_id
    
    extended_names = args.training_classes + ['Unknown']

    num_known = len(args.training_classes)
    short_names = [str(i) for i in range(num_known)] + ['Unk']
    
    target_ids_extended = list(range(num_known)) + [unknown_cls_id]
    
    cm = confusion_matrix(open_set_labels, open_set_preds, labels=target_ids_extended)

    unknown_row = cm[-1, :] 
    total_unknown_samples = np.sum(unknown_row)
    correctly_detected = unknown_row[-1]

    current_open_set_acc = accuracy_score(open_set_labels, open_set_preds)

    if best_acc <= current_open_set_acc:
        best_acc = current_open_set_acc
        
        plt.figure(figsize=(5, 4.5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=short_names,
                    yticklabels=short_names,
                    linewidths=0.5, linecolor='black',  
                    annot_kws={"size": 12}, 
                    cbar_kws={"shrink": 0.85}) 
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.xticks(fontsize=11, rotation=0)
        plt.yticks(fontsize=11, rotation=0)
        
        plt.tight_layout() 
        plt.savefig(os.path.join(args.log_path, 'confusion_matrix.png'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
    
    print("\n=== Open Set Classification Report ===")
    print(classification_report(open_set_labels, open_set_preds, 
                                labels=target_ids_extended,
                                target_names=extended_names, 
                                digits=4,
                                zero_division=0))
    
    print("\n=== Unknown Class Breakdown ===")
    print(f"Total Unknown Samples: {total_unknown_samples}")
    print(f"Correctly Detected as 'Unknown': {correctly_detected} ({correctly_detected/total_unknown_samples:.2%})")
    print("Misclassified as (Leakage):")
    
    for i in range(len(args.training_classes)):
        misclassified_count = unknown_row[i]
        if misclassified_count > 0:
            cls_name = args.training_classes[i]
            ratio = misclassified_count / total_unknown_samples
            print(f"  -> {cls_name}: {misclassified_count} samples ({ratio:.2%})")
    return best_acc