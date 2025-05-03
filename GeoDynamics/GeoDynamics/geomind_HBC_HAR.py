import torch
from geo_mind import GeoMind
from dataset import Imdb_dataset, SeqFCDataset_7tasks, Dataset_Skeleton_seqFC, Dataset_ADNI_seqFC, SeqFCDataset_WM, Dataset_WM, Dataset_ADNI_seq, Dataset_OASIS_seq,Dataset_OASIS_seqFC, Dataset_PPMI_seq,Dataset_PPMI_seqFC, Dataset_HCPA_seq
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.nn as nn
from optimizer import StiefelMetaOptimizer
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import defaultdict
label_attention_map = defaultdict(list)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_data(file_path):
    data = torch.load(file_path)  # Example for loading a tensor from a file
    return data

# Parse arguments for dataset selection
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ADNI', required=False, help="Choose the dataset")
parser.add_argument('--gpu', type=str,default='cuda:3', required=False, help="Choose the gpu")
parser.add_argument('--batch', type=int,default=8, required=False, help="Choose the batch size")
parser.add_argument('--dim', type=int,default=116, required=False, help="Choose the dim")
parser.add_argument('--state', type=int,default=2, required=False, help="Choose the number of clusters")
parser.add_argument('--epoch', type=int,default=300, required=False, help="Choose the number of epochs")
parser.add_argument('--lr', type=float,default=5e-5, required=False, help="Choose the learning rate")
parser.add_argument('--ws', type=int,default=40, required=False, help="Choose the size of window")
parser.add_argument('--con', type=int,default=5, required=False, help="Choose the size of convolution")
parser.add_argument('--expand', type=int,default=1, required=False, help="Choose the size of features")

args = parser.parse_args()
output_dir = '**'
# Dataset loading based on argument
if args.dataset == 'ADNI':
    input_dir = f'**'
    label_dir = '**'
    dataset = Dataset_ADNI_seqFC(input_dir, label_dir)
    time_points = 140
elif args.dataset == 'HCPYA':
    input_dir = f'**'
    label_dir = '**'
    dataset = SeqFCDataset_WM(input_dir, label_dir)
    time_points = 39
elif args.dataset == 'OASIS':
    input_dir = f'**'
    # label_dir = '**'
    dataset = Dataset_OASIS_seqFC(input_dir)
    time_points = 328
elif args.dataset in ['PPMI', 'ABIDE', 'neurocon','taowu']  :
    input_dir = f'**/{args.dataset}'
    # label_dir = '/path/to/OASIS/label'
    dataset = Dataset_PPMI_seqFC(input_dir)
    time_points = 137
elif args.dataset in ['F3D']  :
    input_dir = f'**/{args.dataset}.mat'
    # label_dir = '/path/to/OASIS/label'
    dataset = Dataset_Skeleton_seqFC(input_dir,5)
    time_points = 35-args.ws+1 #state9 dim42
elif args.dataset in ['HDM14','HDM65']  :
    input_dir = f'**/{args.dataset}.mat'
    # label_dir = '/path/to/OASIS/label'
    dataset = Dataset_Skeleton_seqFC(input_dir,args.ws)
    time_points = 901-args.ws+1 #state29 dim90
elif args.dataset in ['UTK']  :
    input_dir = f'**/{args.dataset}.mat'
    # label_dir = '/path/to/OASIS/label'
    dataset = Dataset_Skeleton_seqFC(input_dir,args.ws)
    time_points = 74-args.ws+1 #state10 dim57
    print(time_points)
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

dim = args.dim  # Adjust based on your dataset
d_state = args.state 
d_conv = 5
expand = 1


class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(d_model, dim)
        self.fc2 = nn.Linear(dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        x = self.fc1(hidden_states)
        x = self.relu(x)
        x = self.fc2(x)
        return x
class DualOptimizerWrapper:
    def __init__(self, backbone, classification_head, lr=5e-4):
        # Optimizer for backbone with Stiefel manifold constraint
        opt_backbone = torch.optim.Adam(backbone.parameters(), lr=lr)
        self.optimizer_backbone = StiefelMetaOptimizer(opt_backbone)

        # Optimizer for classification head (standard)
        self.optimizer_head = torch.optim.Adam(classification_head.parameters(), lr=lr)

    def zero_grad(self):
        self.optimizer_backbone.zero_grad()
        self.optimizer_head.zero_grad()

    def step(self):
        self.optimizer_backbone.step()
        self.optimizer_head.step()

    def state_dict(self):
        return {
            'backbone': self.optimizer_backbone.state_dict(),
            'head': self.optimizer_head.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.optimizer_backbone.load_state_dict(state_dict['backbone'])
        self.optimizer_head.load_state_dict(state_dict['head'])

kf = KFold(n_splits=10, shuffle=False, random_state=None)

# Store metrics for each fold
accuracies, precisions, recalls, f1_scores = [], [], [], []
results_per_fold = []

for i, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {i+1}:")

    # Split the dataset into training and testing sets
    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # Initialize Mamba model with classification head
    model = GeoMind(
        d_model=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        time_point=time_points
    ).to(args.gpu)
    print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    classification_head = ClassificationHead(dim, num_classes=args.state).to(args.gpu)

    # Define optimizer and loss function
    optimizer = DualOptimizerWrapper(model, classification_head, lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    classification_head.train()
    for epoch in range((args.epoch)):
        correct = 0
        total = 0
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}", unit="batch"):
            x_batch = batch['input_ids'].to(args.gpu)
            y_batch = batch['labels'].to(args.gpu)

            optimizer.zero_grad()
            outputs, attention = model(x_batch)
            # print(attention.shape)
            mean_matrix_batch = attention.mean(dim=(1))
            # print(mean_matrix.shape)
            for label in y_batch.unique():
                label_mask = (y_batch == label)
                label_attention_map[label.item()].append(mean_matrix_batch[label_mask].mean(dim=0).cpu().detach().numpy())
            # weight_numpy = mean_matrix.cpu().detach().numpy()
            logits = classification_head(outputs.mean(dim=1))
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / total
        print(accuracy)
        if accuracy > 90:
            os.makedirs(output_dir, exist_ok=True)
            for label, attention_matrices in label_attention_map.items():
                mean_matrix = np.mean(attention_matrices, axis=0)
                output_file = f'{args.dataset}_weight_matrix_label_{label}_{accuracy:.2f}.txt'
            
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, 'w') as f:
                    f.write(f"Label: {label}\n")
                    np.savetxt(f, mean_matrix, fmt='%.6f')
                    f.write("\n")
        print(f"Epoch {epoch+1}: Loss = {train_loss/len(train_loader):.4f}, Accuracy = {accuracy:.4f}%")

    # Evaluation loop
    model.eval()
    classification_head.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epoch}", unit="batch"):
            x_batch = batch['input_ids'].to(args.gpu)
            y_batch = batch['labels'].to(args.gpu)
            
            outputs,weight = model(x_batch)
            logits = classification_head(outputs.mean(dim=1))
            _, predicted = torch.max(logits.data, 1)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    results_per_fold.append({
        'fold': i+1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    print(f'Accuracy for fold {i+1}: {accuracy:.2f}%')
    print(f'Precision for fold {i+1}: {precision:.2f}')
    print(f'Recall for fold {i+1}: {recall:.2f}')
    print(f'F1 Score for fold {i+1}: {f1:.2f}')

# Mean and variance of metrics
mean_accuracy = np.mean(accuracies)
variance_accuracy = np.var(accuracies)
mean_precision = np.mean(precisions)
variance_precision = np.var(precisions)
mean_recall = np.mean(recalls)
variance_recall = np.var(recalls)
mean_f1 = np.mean(f1_scores)
variance_f1 = np.var(f1_scores)

print(f'\nMean Accuracy: {mean_accuracy:.2f}%')
print(f'Variance of Accuracy: {variance_accuracy:.2f}%')
print(f'Mean Precision: {mean_precision:.2f}')
print(f'Variance of Precision: {variance_precision:.2f}')
print(f'Mean Recall: {mean_recall:.2f}')
print(f'Variance of Recall: {variance_recall:.2f}')
print(f'Mean F1 Score: {mean_f1:.2f}')
print(f'Variance of F1 Score: {variance_f1:.2f}')

# Save results to a file
with open(f"cross_validation_results_{args.dataset}_geomind{args.batch}_lr{args.lr}_ws{args.ws}.txt", "w") as f:
    for fold_result in results_per_fold:
        f.write(f"Fold {fold_result['fold']}:\n")
        f.write(f"  Accuracy: {fold_result['accuracy']:.2f}%\n")
        f.write(f"  Precision: {fold_result['precision']:.2f}\n")
        f.write(f"  Recall: {fold_result['recall']:.2f}\n")
        f.write(f"  F1 Score: {fold_result['f1']:.2f}\n")
    
    f.write(f'\nMean Accuracy: {mean_accuracy:.2f}%\n')
    f.write(f'Variance of Accuracy: {variance_accuracy:.2f}%\n')
    f.write(f'Mean Precision: {mean_precision:.2f}\n')
    f.write(f'Variance of Precision: {variance_precision:.2f}\n')
    f.write(f'Mean Recall: {mean_recall:.2f}\n')
    f.write(f'Variance of Recall: {variance_recall:.2f}\n')
    f.write(f'Mean F1 Score: {mean_f1:.2f}\n')
    f.write(f'Variance of F1 Score: {variance_f1:.2f}\n')
