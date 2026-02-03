import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pathlib


#Parse arguments from terminal
parser = argparse.ArgumentParser(description='Run method on files.')

parser.add_argument('--output_dir', default="results", type=str, help='Output directory')
parser.add_argument('--name', default="tissue", type=str, help='Dataset name')
parser.add_argument('--preprocessing.meta', type=str, help='Preprocessed metadata')
parser.add_argument('--ScoringTools.fullNES', type=str, help='Tools output collector')
parser.add_argument('--input_type', type=str, help='Tools of origin for input to the script for metrics computation')

args, _ = parser.parse_known_args()

dataset_name = getattr(args, 'name')

input_filepath = getattr(args, 'ScoringTools.fullNES')

output_dir = getattr(args, 'output_dir')
metadata = getattr(args, "preprocessing.meta")

#Create output directory
os.makedirs(f"./{output_dir}", exist_ok=True)

#Load and preprocess metadata
metadata_df = pd.read_csv(metadata, sep='\t')
metadata_df["true_label"] = metadata_df["true_label"].str.upper()

#Make list of all label categories (e.g. tissues, sex, cell cycle phase etc.) collected from metadata
label_types_samples = list(set(metadata_df["true_label"]))
label_types_samples = [label_class.upper() for label_class in label_types_samples if isinstance(label_class, str)]

#Load in parameters dict #Luca same as just above
parameters_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(input_filepath)), "parameters_dict.tsv"), delimiter="\t")
parameters_connect_dict = dict()
methods_connect_dict = dict()

for i, row in parameters_df.iterrows():
    hash_value = row["base_path"].split("/")[-1]
    human_value = row["alias_path"].split("-")[-1]
    parameters_connect_dict[hash_value] = human_value
    human_value = row["alias_path"].split("-")[1].split("_")[0]
    methods_connect_dict[hash_value] = human_value

#Extract specific tool information
filepath_split = input_filepath.split("/")

module = filepath_split[-3]
parameter = filepath_split[-2]

if module == 'gsvaalone':
    module = methods_connect_dict[parameter]

if parameter != "default":
    parameter = parameters_connect_dict[parameter]

#Store tool name in two different versions for later use
tool = module + "/" + parameter
tool_ = module + "_" + parameter

print(tool)

def process_NES_data():
    """Process NES data and prepare samples with ground truth labels."""

    NES_data_df = pd.read_csv(input_filepath, sep='\t')
    NES_data_df["GOI_Set"] = NES_data_df['GOI_Set'].str.upper()

    #List of gene signatures
    gene_signatures = list(NES_data_df["GOI_Set"])

    #Get shared label classes between samples and gene signatures
    geneset_types_shared = set(label_types_samples).intersection(set(gene_signatures))
    metadata_samples_signatures = metadata_df[metadata_df['true_label'].str.upper().isin(geneset_types_shared)]

    #Initialize
    samples_data_dict = {}

    #Add samples with true labels
    for i, row in metadata_samples_signatures.iterrows():
        samples_data_dict[row['filename']] = {}
        samples_data_dict[row['filename']]['true_label'] = row['true_label']

    #Set index and filter to shared gene sets
    NES_data_df = NES_data_df.set_index("GOI_Set")
    NES_data_df = NES_data_df.loc[list(geneset_types_shared)]

    #Add NES scores for each sample
    for sample_name in list(samples_data_dict.keys()):
        if sample_name in NES_data_df.columns:
            # Store all NES scores as a dictionary
            nes_scores = NES_data_df[sample_name].to_dict()
            samples_data_dict[sample_name]['nes_scores'] = nes_scores
        else:
            #Remove samples without NES data
            del samples_data_dict[sample_name]

    print(f"Samples with both ground truth and NES scores: {len(samples_data_dict)}")

    return samples_data_dict, geneset_types_shared, NES_data_df


def calculate_auc_metrics(samples_data_dict, geneset_types_shared, tool_, dataset_name):
    """Calculate AUC metrics for the tool: both label class-wise and overall as macro and weighted AUC."""
    
    #Initialize
    y_true = []
    sample_names = []
    
    for sample_name, data in samples_data_dict.items():
        if 'nes_scores' in data:
            y_true.append(data['true_label'])
            sample_names.append(sample_name)
            
    
    if len(list(set(y_true))) == 1:
        #Get true label distribution
        true_label_dist = pd.Series(y_true).value_counts()
        
        # --- Save overall metrics ---
        overall_metrics_df = pd.DataFrame({
            'Tool': [tool_],
            'AUC_Macro': np.nan, 
            'AUC_Weighted': np.nan})
        overall_metrics_df.to_csv(f'{output_dir}/{dataset_name}-aucmetrics.tsv', sep='\t', index=False)
        
        # --- Save per-label-class metrics: empty df in this case ---
        label_metrics_df = pd.DataFrame()
        label_metrics_df.to_csv(f'{output_dir}/{dataset_name}-aucmetrics_label_classes.tsv', sep='\t', index=False)
    
    else: 
        #Get unique classes
        classes = sorted(list(geneset_types_shared))
        n_classes = len(classes)
        
        #Create binary labels to use for label class-specific AUC computation
        #dim: sample x class_labels (1 if sample belongs to class, else 0)
        y_true_binary = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            #Expand binary cases to 2 columns instead of 1
            y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
        
        #Get NES scores for each class and sample
        y_scores = np.zeros((len(sample_names), n_classes))
        for i, sample_name in enumerate(sample_names):
            nes_dict = samples_data_dict[sample_name]['nes_scores']
            for j, class_name in enumerate(classes):
                y_scores[i, j] = nes_dict.get(class_name, np.nan)
        
        #Handle NaN values
        if np.isnan(y_scores).any():
            print(f"Warning: {np.isnan(y_scores).sum()} NaN values found in NES scores")
            mean_val = np.nanmean(y_scores)
            y_scores = np.nan_to_num(y_scores, nan=mean_val) #Assign NaNs to be the average NES score for a sample
    
            #y_scores = np.nan_to_num(y_scores, nan=np.nanmin(y_scores)) #Assign NaNs to be the minimum NES score for a sample
        
        ###Calculate per-class ROC curves and AUC###
    
        #Initialize
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        roc_failed = False
        
        for i, class_name in enumerate(classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except Exception as e:
                print(y_true_binary)
                n_samples = int(np.sum(y_true_binary[:, i]))
                roc_auc[i] = np.nan
                print(f"ROC calculation failed for {class_name} with sample size {n_samples}): {e}")
                roc_failed = True
    
        #If AUC calculations are problematic, return only nan values. 
        if roc_failed:
            # Create placeholder DataFrames with NA values
            overall_metrics_df = pd.DataFrame({
                'Tool': [tool_],
                'AUC_Macro': [np.nan],
                'AUC_Weighted': [np.nan]
            })
            overall_metrics_df.to_csv(f'{output_dir}/{dataset_name}-aucmetrics.tsv', 
                                     sep='\t', index=False)
            
            true_label_dist = pd.Series(y_true).value_counts()
            label_metrics_rows = []
            for i, class_name in enumerate(classes):
                label_metrics_rows.append({
                    'Label_Class': class_name,
                    'Count': true_label_dist.get(class_name, 0),
                    'AUC': roc_auc[i]
                })
            label_metrics_df = pd.DataFrame(label_metrics_rows)
            label_metrics_df.to_csv(f'{output_dir}/{dataset_name}-aucmetrics_label_classes.tsv', 
                                   sep='\t', index=False)
            
            return overall_metrics_df, label_metrics_df
        
        # ---- Calculate macro-average ROC curve and AUC ----
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        try:
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = roc_auc_score(y_true_binary, y_scores, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate macro AUC: {e}")
            roc_auc["macro"] = np.nan
    
        # ---- Calculate weighted-average ROC curve ----
        class_counts = np.sum(y_true_binary, axis=0)
        weights = class_counts / np.sum(class_counts)
    
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        weighted_tpr = np.zeros_like(all_fpr)
    
        for i in range(n_classes):
            weighted_tpr += weights[i] * np.interp(all_fpr, fpr[i], tpr[i])
    
        #Calculate weighted AUC
        try:
            fpr["weighted"] = all_fpr
            tpr["weighted"] = weighted_tpr
            roc_auc["weighted"] = roc_auc_score(y_true_binary, y_scores, average='weighted', multi_class='ovr')
    
            #Sanity check
            # If all classes have equal sample counts, macro and weighted should be approximately equal
            if len(set(class_counts)) == 1 and not np.isnan(roc_auc["macro"]):  # All counts are the same
                if not np.isclose(roc_auc["macro"], roc_auc["weighted"], rtol=1e-3):
                    print(f"Warning: Macro ({roc_auc['macro']:.6f}) and weighted ({roc_auc['weighted']:.6f})"
                        f"AUC differ despite balanced classes. This may indicate a calculation issue.")
                    
        except Exception as e:
            print(f"Warning: Could not calculate weighted AUC: {e}")
            roc_auc["weighted"] = np.nan
        
        #Get true label distribution
        true_label_dist = pd.Series(y_true).value_counts()
        
        # --- Save overall metrics ---
        overall_metrics_df = pd.DataFrame({
            'Tool': [tool_],
            'AUC_Macro': [round(roc_auc["macro"], 3)],
            'AUC_Weighted': [round(roc_auc["weighted"], 3)]
        })
        overall_metrics_df.to_csv(f'{output_dir}/{dataset_name}-aucmetrics.tsv', sep='\t', index=False)
        
        # --- Save per-label-class metrics ---
        label_metrics_rows = []
        for i, class_name in enumerate(classes):
            label_metrics_rows.append({
                'Label_Class': class_name,
                'Count': true_label_dist.get(class_name, 0),
                'AUC': round(roc_auc[i], 3)
            })
        
        label_metrics_df = pd.DataFrame(label_metrics_rows)
        label_metrics_df.to_csv(f'{output_dir}/{dataset_name}-aucmetrics_label_classes.tsv', sep='\t', index=False)
        
        #Create ROC plot
        plot_roc_curves(fpr, tpr, roc_auc, classes, tool_, tool, dataset_name)
    

def plot_roc_curves(fpr, tpr, roc_auc, classes, tool_, tool, dataset_name):
    """
    Plot ROC curves for all label class in one plot.
    """
    
    n_classes = len(classes)
    
    plt.figure(figsize=(12, 10))
    
    #Plot per-class ROC curves
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.7,
                label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    if not np.isnan(roc_auc["macro"]):
        #Plot macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                color='navy', linestyle='--', linewidth=2.5,
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    if not np.isnan(roc_auc["weighted"]):
        #plot weighted-average ROC curve
        plt.plot(fpr["weighted"], tpr["weighted"],
                color='navy', linestyle=':', linewidth=2.5,
                label=f'Weighted average (AUC = {roc_auc["weighted"]:.3f})')
    
    #Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves (Tool: {tool}, dataset: {dataset_name})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    #Save plot
    plt.savefig(f"./{output_dir}/{dataset_name}-ROC_{tool_}.png", dpi=300, bbox_inches='tight')
    plt.close()
    

#Main
samples_data_dict, geneset_types_shared, NES_data_df = process_NES_data()
calculate_auc_metrics(samples_data_dict, geneset_types_shared, tool_, dataset_name)

pathlib.Path(f"{output_dir}/{dataset_name}-metrics_complete.flag").touch()
