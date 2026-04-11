from exp.exp_basic import Exp_Basic
from models.gm_cnf import GM_CNF
from data.data_loader import get_loaders 
from utils.tools import EarlyStopping, get_adaptive_threshold
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import itertools

from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from utils.tools import plot_tsne, plot_nll_distribution, evaluate_open_set_and_plot_cm

class Exp_GM_CNF(Exp_Basic):
    def __init__(self, args):
        super(Exp_GM_CNF, self).__init__(args)
        self.count = 0

    def _build_model(self):
        return GM_CNF(self.args)

    def train(self):
        labeled_loader, unlabeled_loader, vali_loader, test_loader = get_loaders(
            self.args.dataset,
            batch_size=self.args.batch_size,
            win_len=self.args.window_size,
            stride=self.args.stride,
            few_shot_num=self.args.few_shot_num,
            training_classes=self.args.training_classes,
            testing_classes=self.args.testing_classes
        )
        path = self.args.log_path
        writer = SummaryWriter(log_dir=path)
        print(f"TensorBoard log directory: {path}")
        
        early_stopping = EarlyStopping(patience=5, verbose=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        print(f"Start training with {self.args.num_classes} clusters.")
        
        if self.args.checkpoint is not None:
            print(f"Loading checkpoint from {self.args.checkpoint}...")
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = []
            sup_loss_list = []
            unsup_loss_list = []

            labeled_iter = itertools.cycle(labeled_loader)
            
            for i, (x_u, c_u, _) in enumerate(unlabeled_loader):
                x_u = x_u.to(self.device)
                c_u = c_u.to(self.device)

                x_l, c_l, y_l = next(labeled_iter)
                x_l = x_l.to(self.device)
                c_l = c_l.to(self.device)
                y_l = y_l.to(self.device).long()

                noise_level = 0.001
                x_l = x_l + torch.randn_like(x_l) * noise_level
                x_u = x_u + torch.randn_like(x_u) * noise_level
                
                optimizer.zero_grad()
                loss_sup = self.model.compute_loss(x_l, c_l, label=y_l)

                if getattr(self.args, 'ablation_pl', False):
                    self.model.eval()
                    with torch.no_grad():
                        pseudo_labels, _, _ = self.model.predict_dist(x_u, c_u)
                    self.model.train()
                    loss_unsup = self.model.compute_loss(x_u, c_u, label=pseudo_labels)
                else:
                    loss_unsup = self.model.compute_loss(x_u, c_u, label=None)

                w_sup = 1.0
                w_unsup = 0.5 
                
                loss = w_sup * loss_sup + w_unsup * loss_unsup
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss.append(loss.item())
                sup_loss_list.append(loss_sup.item())
                unsup_loss_list.append(loss_unsup.item())
            
            avg_loss = np.average(train_loss)
            avg_sup = np.average(sup_loss_list)
            avg_unsup = np.average(unsup_loss_list)
            
            print(f"\nEpoch {epoch+1} | Total: {avg_loss:.4f} | sup: {avg_sup:.4f} | unsup: {avg_unsup:.4f}")
            writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
            writer.add_scalar('Train/Cls_Loss', avg_sup, epoch)
            writer.add_scalar('Train/Flow_Loss', avg_unsup, epoch)
            
            # === Validation ===
            self.model.eval()
            val_loss_list = []
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for x, c, y in vali_loader:
                    x = x.to(self.device)
                    c = c.to(self.device)
                    y = y.to(self.device).long()

                    val_loss = self.model.compute_loss(x, c, label=y)
                    val_loss_list.append(val_loss.item())

                    preds, _, _ = self.model.predict_dist(x, c)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            
            avg_val_loss = np.average(val_loss_list)

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}")
            
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_acc, epoch)

            self.test(test_loader=test_loader, epoch=epoch)

            early_stopping(avg_val_loss, self.model, path)
            torch.save(self.model.state_dict(), os.path.join(path, f"checkpoint_{epoch+1}.pth"))
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded from", best_model_path)

    def test(self, test_loader=None, epoch=None):
        if test_loader is None:
            _, _, _, test_loader = get_loaders(
                self.args.dataset, 
                batch_size=self.args.batch_size,
                win_len=self.args.window_size,
                stride=self.args.stride,
                few_shot_num=3,
                training_classes=self.args.training_classes,
                testing_classes=self.args.testing_classes
            )
        
        if self.args.checkpoint is not None:
            print(f"Loading checkpoint from {self.args.checkpoint} for testing...")
            self.model.load_state_dict(torch.load(self.args.checkpoint))

        self.model.eval()
        
        preds_list = []
        scores_list = []
        true_labels = []
        features_list = []
        rpm_list = []
        
        print("Starting Testing...")
        with torch.no_grad():
            for x, c, y in test_loader:
                x = x.to(self.device)
                c = c.to(self.device)

                preds, nll, z = self.model.predict_dist(x, c)
                
                preds_list.append(preds.cpu().numpy())
                scores_list.append(nll.cpu().numpy())
                true_labels.append(y.numpy())
                features_list.append(z.cpu().numpy())
                rpm_list.append(c[:, 0, 0].cpu().numpy())
                
        all_preds = np.concatenate(preds_list)
        all_scores = np.concatenate(scores_list)
        all_labels = np.concatenate(true_labels)
        all_features = np.concatenate(features_list)
        all_rpms = np.concatenate(rpm_list)

        known_class_ids = list(range(self.args.num_known_classes))
        
        # ----------------------------------------------------
        # Task 1: Known Class Classification Accuracy
        # ----------------------------------------------------
        known_mask = np.isin(all_labels, known_class_ids)
        unknown_mask = ~known_mask
        
        if epoch % 5 == 0:
            plot_tsne(self.args, all_features, all_labels, known_mask, unknown_mask, all_rpms, epoch=epoch)

        if np.sum(known_mask) > 0:
            print("\n=== Task 1: Classification on Known Classes ===")
            y_true_known = all_labels[known_mask]
            y_pred_known = all_preds[known_mask]
            
            acc = accuracy_score(y_true_known, y_pred_known)
            print(f"Known Class Accuracy: {acc:.2%}")
            target_ids = list(range(len(self.args.training_classes)))
            print(classification_report(y_true_known, y_pred_known, 
                                      labels=target_ids, 
                                      target_names=self.args.training_classes,
                                      digits=4, zero_division=0))
        
        # ----------------------------------------------------
        # Task 2: Unknown Class Detection (Open Set Recognition)
        # ----------------------------------------------------
        
        if np.sum(unknown_mask) > 0:
            print("\n=== Task 2: Unknown Class Detection (Distance Based) ===")
            dist_known = all_scores[known_mask]
            dist_unknown = all_scores[unknown_mask]
            
            print(f"Avg Dist (Known): {np.mean(dist_known):.4f}")
            print(f"Avg Dist (Unknown): {np.mean(dist_unknown):.4f}")

            ood_labels = np.zeros_like(all_scores)
            ood_labels[unknown_mask] = 1 
            
            auroc = roc_auc_score(ood_labels, all_scores)
            print(f"AUROC: {auroc:.4f}")

        # ==========================================
        # Task 3: Open Set Detailed Evaluation
        # ==========================================
        if np.sum(known_mask) > 0 and np.sum(unknown_mask) > 0:
            print("\n=== Task 3: Open Set Detailed Evaluation ===")

            dist_known = all_scores[known_mask]
            threshold, method_name = get_adaptive_threshold(dist_known, method="iqr")
            print(f"Calculated Threshold: {threshold:.4f}")
            plot_nll_distribution(self.args, dist_known, dist_unknown, threshold, epoch)

            self.count = evaluate_open_set_and_plot_cm(
                args=self.args, 
                all_scores=all_scores, 
                all_labels=all_labels, 
                all_preds=all_preds, 
                known_mask=known_mask, 
                unknown_mask=unknown_mask, 
                threshold=threshold, 
                best_acc=self.count
            )