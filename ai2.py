import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                           roc_auc_score, classification_report, roc_curve, 
                           precision_recall_curve, f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import kagglehub
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

class EnhancedRF_XGB_Comparison:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def preprocess_features(self):
        """Preprocess categorical and numerical features"""
        print("Preprocessing categorical and numerical features...")
        
        # Separate categorical and numerical columns
        categorical_columns = []
        numerical_columns = []
        
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        
        print(f"Found {len(categorical_columns)} categorical and {len(numerical_columns)} numerical columns")
        
        # Handle categorical columns with Label Encoding
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            # Handle missing values by filling with 'unknown'
            self.X[col] = self.X[col].fillna('unknown')
            # Remove leading/trailing whitespace
            self.X[col] = self.X[col].astype(str).str.strip()
            self.X[col] = le.fit_transform(self.X[col])
            label_encoders[col] = le
        
        # Handle numerical columns - fill missing with median
        for col in numerical_columns:
            median_val = self.X[col].median()
            self.X[col] = self.X[col].fillna(median_val)
        
        # Convert all columns to float for consistency
        self.X = self.X.astype(float)
        
        print("Feature preprocessing completed")
        
    def load_data(self):
        """Load Adult Census Income dataset from Kaggle"""
        try:
            print("Downloading Adult Census Income dataset from Kaggle...")
            
            # Download the dataset
            path = kagglehub.dataset_download("uciml/adult-census-income")
            print(f"Dataset downloaded to: {path}")
            
            # Load the data - Adult dataset typically has a specific structure
            import os
            data_files = os.listdir(path)
            print(f"Available files: {data_files}")
            
            # Look for the adult.data file (most common naming)
            data_file = None
            for file in data_files:
                if 'adult' in file.lower() and ('.data' in file or '.csv' in file):
                    data_file = os.path.join(path, file)
                    break
            
            if data_file is None:
                # If no specific file found, use the first CSV file
                csv_files = [f for f in data_files if f.endswith('.csv')]
                if csv_files:
                    data_file = os.path.join(path, csv_files[0])
                else:
                    raise FileNotFoundError("No suitable data file found")
            
            print(f"Loading data from: {data_file}")
            
            # Adult dataset column names
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            
            # Try to load with different delimiters
            try:
                if data_file.endswith('.csv'):
                    data = pd.read_csv(data_file)
                else:
                    # Try comma first, then space
                    try:
                        data = pd.read_csv(data_file, names=column_names, skipinitialspace=True)
                    except:
                        data = pd.read_csv(data_file, names=column_names, sep=r'\s*,\s*', engine='python')
            except Exception as e:
                print(f"Error reading file with standard method: {e}")
                # Fallback: try reading without headers and assign column names
                data = pd.read_csv(data_file, header=None, skipinitialspace=True)
                if data.shape[1] == len(column_names):
                    data.columns = column_names
                else:
                    print(f"Data has {data.shape[1]} columns, expected {len(column_names)}")
                    # Use generic column names
                    data.columns = [f'feature_{i}' for i in range(data.shape[1]-1)] + ['target']
            
            print(f"Raw dataset shape: {data.shape}")
            print("First few rows:")
            print(data.head())
            
            # Identify target column
            target_col = None
            for col in data.columns:
                if 'income' in col.lower() or col == data.columns[-1]:
                    target_col = col
                    break
            
            if target_col is None:
                target_col = data.columns[-1]  # Assume last column is target
            
            print(f"Target column identified: {target_col}")
            print(f"Target value counts:\n{data[target_col].value_counts()}")
            
            # Prepare features and target
            self.X = data.drop(columns=[target_col])
            self.y = data[target_col]
            
            # Clean target values (remove whitespace and normalize)
            self.y = self.y.astype(str).str.strip()
            
            # Convert target to binary (0/1)
            unique_targets = self.y.unique()
            print(f"Unique target values: {unique_targets}")
            
            # Map target values to binary
            if len(unique_targets) == 2:
                # Typically '>50K' and '<=50K' or similar
                positive_class = [val for val in unique_targets if '>' in val or 'high' in val.lower()]
                if positive_class:
                    positive_class = positive_class[0]
                else:
                    positive_class = unique_targets[1]  # Second value as positive
                
                self.y = (self.y == positive_class).astype(int)
                print(f"Target mapped: '{positive_class}' -> 1, others -> 0")
            else:
                raise ValueError(f"Expected 2 classes, found {len(unique_targets)}")
            
            # Process features
            print("Processing features...")
            self.preprocess_features()
            
            print(f"Final dataset shape: {self.X.shape}")
            print(f"Final target distribution:\n{pd.Series(self.y).value_counts()}")
            
            # Calculate class imbalance
            class_counts = pd.Series(self.y).value_counts()
            imbalance_ratio = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1
            print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def advanced_preprocessing(self):
        """Advanced preprocessing pipeline"""
        print("\nAdvanced preprocessing...")
        
        # Store original feature names
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(self.X.shape[1])]
        
        # Advanced imputation with KNN
        print("Using KNN imputer...")
        imputer = KNNImputer(n_neighbors=5)
        self.X = pd.DataFrame(imputer.fit_transform(self.X))
        
        # Remove constant and near-constant columns
        constant_cols = []
        for col in self.X.columns:
            if self.X[col].nunique() <= 1:
                constant_cols.append(col)
            elif self.X[col].std() < 1e-6:  # Near constant
                constant_cols.append(col)
        
        if constant_cols:
            self.X = self.X.drop(columns=constant_cols)
            print(f"Removed {len(constant_cols)} constant/near-constant columns")
        
        # Remove highly correlated features with threshold 0.95
        corr_matrix = self.X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_cols = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > 0.95):
                high_corr_cols.append(column)
        
        if high_corr_cols:
            self.X = self.X.drop(columns=high_corr_cols)
            print(f"Removed {len(high_corr_cols)} highly correlated columns")
        
        print(f"Final feature shape after preprocessing: {self.X.shape}")
    
    def split_data(self):
        """Split data with stratification"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\nData split:")
        print(f"Training: {self.X_train.shape[0]} samples, Positive class: {self.y_train.sum()} ({self.y_train.mean():.2%})")
        print(f"Test: {self.X_test.shape[0]} samples, Positive class: {self.y_test.sum()} ({self.y_test.mean():.2%})")
    
    def prepare_datasets_with_sampling(self):
        """Prepare different versions of datasets with various sampling techniques"""
        self.datasets = {}
        
        # 1. Original (no sampling)
        self.datasets['Original'] = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'description': 'No sampling applied'
        }
        
        # 2. ADASYN oversampling
        try:
            adasyn = ADASYN(random_state=42)
            X_adasyn, y_adasyn = adasyn.fit_resample(self.X_train, self.y_train)
            self.datasets['ADASYN'] = {
                'X_train': X_adasyn,
                'y_train': y_adasyn,
                'description': 'ADASYN oversampling'
            }
            print(f"ADASYN sampling: {X_adasyn.shape[0]} samples")
        except Exception as e:
            print(f"ADASYN failed: {e}")
        
        # 3. SMOTE oversampling
        try:
            smote = SMOTE(random_state=42)
            X_smote, y_smote = smote.fit_resample(self.X_train, self.y_train)
            self.datasets['SMOTE'] = {
                'X_train': X_smote,
                'y_train': y_smote,
                'description': 'SMOTE oversampling'
            }
            print(f"SMOTE sampling: {X_smote.shape[0]} samples")
        except Exception as e:
            print(f"SMOTE failed: {e}")
        
        # 4. Random undersampling
        try:
            undersampler = RandomUnderSampler(random_state=42)
            X_under, y_under = undersampler.fit_resample(self.X_train, self.y_train)
            self.datasets['Undersampled'] = {
                'X_train': X_under,
                'y_train': y_under,
                'description': 'Random undersampling'
            }
            print(f"Undersampling: {X_under.shape[0]} samples")
        except Exception as e:
            print(f"Undersampling failed: {e}")
    
    def train_random_forest_variants(self):
        """Train different Random Forest variants"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST VARIANTS")
        print("="*60)
        
        rf_configs = {
            'RF_Default': {
                'params': {'random_state': 42, 'n_jobs': -1},
                'name': 'Random Forest (Default)'
            },
            'RF_Optimized': {
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'name': 'Random Forest (Optimized)'
            },
            'RF_Deep': {
                'params': {
                    'n_estimators': 300,
                    'max_depth': 25,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'name': 'Random Forest (Deep Trees)'
            },
            'RF_ClassWeight': {
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'name': 'Random Forest (Class Weighted)'
            }
        }
        
        for sampling_method, dataset in self.datasets.items():
            X_train_samp = dataset['X_train']
            y_train_samp = dataset['y_train']
            
            # Apply scaling for each dataset
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_samp)
            
            for rf_key, rf_config in rf_configs.items():
                model_key = f"{rf_key}_{sampling_method}"
                
                print(f"Training {rf_config['name']} with {dataset['description']}...")
                
                start_time = time.time()
                
                model = RandomForestClassifier(**rf_config['params'])
                model.fit(X_train_scaled, y_train_samp)
                
                training_time = time.time() - start_time
                
                self.models[model_key] = {
                    'model': model,
                    'scaler': scaler,
                    'name': rf_config['name'],
                    'sampling': dataset['description'],
                    'algorithm': 'Random Forest',
                    'training_time': training_time,
                    'params': rf_config['params']
                }
    
    def train_xgboost_variants(self):
        """Train different XGBoost variants"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST VARIANTS")
        print("="*60)
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        xgb_configs = {
            'XGB_Default': {
                'params': {
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'n_jobs': -1
                },
                'name': 'XGBoost (Default)'
            },
            'XGB_Optimized': {
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'n_jobs': -1
                },
                'name': 'XGBoost (Optimized)'
            },
            'XGB_ClassWeight': {
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'n_jobs': -1
                },
                'name': 'XGBoost (Class Weighted)'
            },
            'XGB_Conservative': {
                'params': {
                    'n_estimators': 300,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'n_jobs': -1
                },
                'name': 'XGBoost (Conservative)'
            }
        }
        
        for sampling_method, dataset in self.datasets.items():
            X_train_samp = dataset['X_train']
            y_train_samp = dataset['y_train']
            
            # Apply scaling for each dataset
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_samp)
            
            for xgb_key, xgb_config in xgb_configs.items():
                model_key = f"{xgb_key}_{sampling_method}"
                
                print(f"Training {xgb_config['name']} with {dataset['description']}...")
                
                start_time = time.time()
                
                model = xgb.XGBClassifier(**xgb_config['params'])
                model.fit(X_train_scaled, y_train_samp)
                
                training_time = time.time() - start_time
                
                self.models[model_key] = {
                    'model': model,
                    'scaler': scaler,
                    'name': xgb_config['name'],
                    'sampling': dataset['description'],
                    'algorithm': 'XGBoost',
                    'training_time': training_time,
                    'params': xgb_config['params']
                }
    
    def optimize_threshold(self, y_true, y_proba, metric='f1'):
        """Find optimal threshold for classification"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                if precision + recall > 0:
                    score = 2 * (precision * recall) / (precision + recall)
                else:
                    score = 0
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        for model_key, model_info in self.models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            print(f"\nEvaluating {model_info['name']} ({model_info['sampling']})...")
            
            # Scale test data
            X_test_scaled = scaler.transform(self.X_test)
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            prediction_time = time.time() - start_time
            
            # Optimize threshold
            best_threshold, best_f1 = self.optimize_threshold(self.y_test, y_pred_proba, 'f1')
            y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)
            
            # Calculate all metrics
            accuracy = accuracy_score(self.y_test, y_pred_optimized)
            precision = precision_score(self.y_test, y_pred_optimized, zero_division=0)
            recall = recall_score(self.y_test, y_pred_optimized, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_optimized, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred_optimized)
            
            # Store results
            self.results[model_key] = {
                'model': model,
                'algorithm': model_info['algorithm'],
                'model_name': model_info['name'],
                'sampling_method': model_info['sampling'],
                'predictions': y_pred_optimized,
                'predictions_proba': y_pred_proba,
                'best_threshold': best_threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'training_time': model_info['training_time'],
                'prediction_time': prediction_time,
                'params': model_info['params'],
                'confusion_matrix': cm
            }
            
            print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    def plot_single_confusion_matrix(self, cm, model_name, sampling_method, algorithm, f1_score, ax):
        """Plot a single confusion matrix with enhanced styling"""
        # Create labels
        labels = ['≤50K', '>50K']
        
        # Create annotation text
        annot_text = np.array([
            [f'{cm[0,0]}\n({cm[0,0]/(cm[0,0]+cm[0,1]):.1%})',
             f'{cm[0,1]}\n({cm[0,1]/(cm[0,0]+cm[0,1]):.1%})'],
            [f'{cm[1,0]}\n({cm[1,0]/(cm[1,0]+cm[1,1]):.1%})',
             f'{cm[1,1]}\n({cm[1,1]/(cm[1,0]+cm[1,1]):.1%})']
        ])
        
        # Create the heatmap
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar=True, square=True, ax=ax,
                   cbar_kws={'shrink': 0.6})
        
        # Customize the plot
        ax.set_title(f'{model_name}\n{sampling_method}\nF1: {f1_score:.3f}', 
                    fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', fontweight='bold', fontsize=9)
        ax.set_ylabel('Actual', fontweight='bold', fontsize=9)
        
        # Add algorithm badge
        algorithm_color = '#FF6B6B' if algorithm == 'Random Forest' else '#4ECDC4'
        ax.text(0.02, 0.98, algorithm, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=algorithm_color, alpha=0.7),
               fontsize=8, fontweight='bold', verticalalignment='top')
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')


     
    def plot_roc_curves_comparison(self):
        """Plot ROC curves for top performing models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get top 5 models by F1 score
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:5]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_key, result) in enumerate(top_models):
            fpr, tpr, _ = roc_curve(self.y_test, result['predictions_proba'])
            ax1.plot(fpr, tpr, color=colors[i], 
                    label=f"{result['model_name'][:15]}... (AUC={result['roc_auc']:.3f})",
                    linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Top 5 Models')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall curves
        for i, (model_key, result) in enumerate(top_models):
            precision, recall, _ = precision_recall_curve(self.y_test, result['predictions_proba'])
            ax2.plot(recall, precision, color=colors[i],
                    label=f"{result['model_name'][:15]}... (F1={result['f1_score']:.3f})",
                    linewidth=2)
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Top 5 Models')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    
    def plot_all_confusion_matrices(self):
        """Plot confusion matrices for all models in organized pages"""
        print("\n" + "="*60)
        print("CREATING CONFUSION MATRICES VISUALIZATION")
        print("="*60)
        
        if not self.results:
            print("No results available for plotting.")
            return
        
        # Sort models by F1 score for better organization
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        # Plot in batches of 6 per page
        models_per_page = 6
        total_models = len(sorted_results)
        total_pages = (total_models + models_per_page - 1) // models_per_page
        
        print(f"Creating {total_pages} pages with {total_models} models total...")
        
        for page in range(total_pages):
            start_idx = page * models_per_page
            end_idx = min(start_idx + models_per_page, total_models)
            page_models = sorted_results[start_idx:end_idx]
            
            # Create figure with subplots
            n_models = len(page_models)
            if n_models <= 3:
                rows, cols = 1, n_models
                figsize = (5 * n_models, 5)
            else:
                rows, cols = 2, 3
                figsize = (1, 8)
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            # Handle single subplot case
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            else:
                axes = axes.flatten()
            
            # Plot confusion matrices
            for i, (model_key, result) in enumerate(page_models):
                self.plot_single_confusion_matrix(
                    result['confusion_matrix'],
                    result['model_name'],
                    result['sampling_method'],
                    result['algorithm'],
                    result['f1_score'],
                    axes[i]
                )
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
            plt.show()
    
    def plot_top_confusion_matrices(self, top_n=8):
        """Plot confusion matrices for top N performing models"""
        print(f"\n{'='*60}")
        print(f"TOP {top_n} MODELS - CONFUSION MATRICES")
        print(f"{'='*60}")
        
        if not self.results:
            print("No results available for plotting.")
            return
        
        # Get top N models by F1 score
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:top_n]
        
        # Create figure
        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        fig.suptitle(f'Top {top_n} Models - Confusion Matrices (Ranked by F1-Score)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        axes = axes.flatten()
        
        # Plot confusion matrices for top models
        for i, (model_key, result) in enumerate(top_models):
            self.plot_single_confusion_matrix(
                result['confusion_matrix'],
                result['model_name'],
                result['sampling_method'],
                result['algorithm'],
                result['f1_score'],
                axes[i]
            )
            
            # Add ranking number
            axes[i].text(0.98, 0.02, f'#{i+1}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='gold', alpha=0.8),
                        fontsize=12, fontweight='bold', ha='right', va='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.25, wspace=0.3)
        plt.show()
        
        # Print ranking table
        print(f"\nTop {top_n} Models Ranking:")
        print("-" * 80)
        for i, (model_key, result) in enumerate(top_models):
            print(f"{i+1:2d}. {result['model_name']} ({result['sampling_method']})")
            print(f"    F1: {result['f1_score']:.4f} | Precision: {result['precision']:.4f} | Recall: {result['recall']:.4f}")
    
    def create_confusion_matrix_summary(self):
        """Create a comprehensive summary of confusion matrix metrics"""
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results available for analysis.")
            return
        
        # Create comprehensive summary
        summary_data = []
        for model_key, result in self.results.items():
            cm = result['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            summary_data.append({
                'Model': result['model_name'][:25] + '...' if len(result['model_name']) > 25 else result['model_name'],
                'Sampling': result['sampling_method'][:15] + '...' if len(result['sampling_method']) > 15 else result['sampling_method'],
                'Algorithm': result['algorithm'],
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                'Sensitivity': result['recall'],
                'Specificity': specificity,
                'PPV': result['precision'],
                'NPV': npv,
                'F1': result['f1_score'],
                'Accuracy': result['accuracy'],
                'AUC': result['roc_auc'],
                'FPR': fpr,
                'FNR': fnr
            })
        
        # Create DataFrame and sort by F1 score
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('F1', ascending=False)
        
        # Print summary table
        print("\nDetailed Confusion Matrix Metrics:")
        print("-" * 120)
        print(f"{'Model':<25} {'Sampling':<15} {'Algo':<12} {'TP':<4} {'TN':<5} {'FP':<4} {'FN':<4} {'F1':<6} {'Acc':<6} {'Sens':<6} {'Spec':<6}")
        print("-" * 120)
        
        for _, row in df_summary.iterrows():
            print(f"{row['Model']:<25} {row['Sampling']:<15} {row['Algorithm']:<12} "
                  f"{row['TP']:<4} {row['TN']:<5} {row['FP']:<4} {row['FN']:<4} "
                  f"{row['F1']:<6.3f} {row['Accuracy']:<6.3f} {row['Sensitivity']:<6.3f} {row['Specificity']:<6.3f}")
        
        return df_summary
    
    def plot_performance_vs_confusion_metrics(self):
        """Plot performance metrics derived from confusion matrices"""
        print("\n" + "="*60)
        print("PERFORMANCE VS CONFUSION METRICS ANALYSIS")
        print("="*60)
        
        if not self.results:
            print("No results available for plotting.")
            return
        
        # Extract metrics for plotting
        metrics_data = []
        for model_key, result in self.results.items():
            cm = result['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            metrics_data.append({
                'model_key': model_key,
                'algorithm': result['algorithm'],
                'sampling': result['sampling_method'],
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall'],
                'specificity': specificity,
                'fpr': fpr,
                'fnr': fnr,
                'accuracy': result['accuracy'],
                'roc_auc': result['roc_auc']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Precision vs Recall
        for algo in df_metrics['algorithm'].unique():
            algo_data = df_metrics[df_metrics['algorithm'] == algo]
            color = '#FF6B6B' if algo == 'Random Forest' else '#4ECDC4'
            axes[0,0].scatter(algo_data['recall'], algo_data['precision'], 
                            label=algo, alpha=0.7, s=60, c=color)
        
        axes[0,0].set_xlabel('Recall (Sensitivity)')
        axes[0,0].set_ylabel('Precision (PPV)')
        axes[0,0].set_title('Precision vs Recall')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Sensitivity vs Specificity
        for algo in df_metrics['algorithm'].unique():
            algo_data = df_metrics[df_metrics['algorithm'] == algo]
            color = '#FF6B6B' if algo == 'Random Forest' else '#4ECDC4'
            axes[0,1].scatter(algo_data['specificity'], algo_data['recall'], 
                            label=algo, alpha=0.7, s=60, c=color)
        
        axes[0,1].set_xlabel('Specificity')
        axes[0,1].set_ylabel('Sensitivity (Recall)')
        axes[0,1].set_title('Sensitivity vs Specificity')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. F1 Score vs ROC-AUC
        for algo in df_metrics['algorithm'].unique():
            algo_data = df_metrics[df_metrics['algorithm'] == algo]
            color = '#FF6B6B' if algo == 'Random Forest' else '#4ECDC4'
            axes[0,2].scatter(algo_data['roc_auc'], algo_data['f1_score'], 
                            label=algo, alpha=0.7, s=60, c=color)
        
        axes[0,2].set_xlabel('ROC-AUC')
        axes[0,2].set_ylabel('F1-Score')
        axes[0,2].set_title('F1-Score vs ROC-AUC')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. False Positive Rate vs False Negative Rate
        for algo in df_metrics['algorithm'].unique():
            algo_data = df_metrics[df_metrics['algorithm'] == algo]
            color = '#FF6B6B' if algo == 'Random Forest' else '#4ECDC4'
            axes[1,0].scatter(algo_data['fpr'], algo_data['fnr'], 
                            label=algo, alpha=0.7, s=60, c=color)
        
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('False Negative Rate')
        axes[1,0].set_title('Error Rates Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Sampling Method Performance
        sampling_perf = df_metrics.groupby('sampling')['f1_score'].mean().sort_values(ascending=True)
        axes[1,1].barh(range(len(sampling_perf)), sampling_perf.values, 
                      color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
        axes[1,1].set_yticks(range(len(sampling_perf)))
        axes[1,1].set_yticklabels(sampling_perf.index)
        axes[1,1].set_xlabel('Average F1-Score')
        axes[1,1].set_title('Sampling Method Performance')
        axes[1,1].grid(True, alpha=0.3, axis='x')
        
        # 6. Algorithm Comparison Box Plot
        rf_f1 = df_metrics[df_metrics['algorithm'] == 'Random Forest']['f1_score']
        xgb_f1 = df_metrics[df_metrics['algorithm'] == 'XGBoost']['f1_score']
        
        box_data = [rf_f1, xgb_f1]
        box_labels = ['Random Forest', 'XGBoost']
        
        bp = axes[1,2].boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1,2].set_ylabel('F1-Score')
        axes[1,2].set_title('Algorithm Performance Distribution')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_comprehensive_comparison_table(self):
        """Create a comprehensive comparison table"""
        print("\n" + "="*120)
        print("COMPREHENSIVE MODEL COMPARISON TABLE")
        print("="*120)
        
        # Create comparison dataframe
        comparison_data = []
        for model_key, result in self.results.items():
            comparison_data.append({
                'Algorithm': result['algorithm'],
                'Model_Variant': result['model_name'],
                'Sampling_Method': result['sampling_method'],
                'F1_Score': result['f1_score'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Accuracy': result['accuracy'],
                'ROC_AUC': result['roc_auc'],
                'Training_Time': result['training_time'],
                'Prediction_Time': result['prediction_time'],
                'Threshold': result['best_threshold']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by F1 score
        df_comparison_sorted = df_comparison.sort_values('F1_Score', ascending=False)
        
        print(df_comparison_sorted.to_string(index=False, float_format='%.4f'))
        
        return df_comparison_sorted
    
    def plot_algorithm_performance_comparison(self):
        """Plot performance comparison between RF and XGBoost"""
        if not self.results:
            print("No results to plot.")
            return
        
        # Separate RF and XGBoost results
        rf_results = {k: v for k, v in self.results.items() if v['algorithm'] == 'Random Forest'}
        xgb_results = {k: v for k, v in self.results.items() if v['algorithm'] == 'XGBoost'}
        
        # Calculate average performance for each algorithm
        rf_metrics = {
            'F1': np.mean([r['f1_score'] for r in rf_results.values()]),
            'Precision': np.mean([r['precision'] for r in rf_results.values()]),
            'Recall': np.mean([r['recall'] for r in rf_results.values()]),
            'ROC_AUC': np.mean([r['roc_auc'] for r in rf_results.values()]),
            'Accuracy': np.mean([r['accuracy'] for r in rf_results.values()])
        }
        
        xgb_metrics = {
            'F1': np.mean([r['f1_score'] for r in xgb_results.values()]),
            'Precision': np.mean([r['precision'] for r in xgb_results.values()]),
            'Recall': np.mean([r['recall'] for r in xgb_results.values()]),
            'ROC_AUC': np.mean([r['roc_auc'] for r in xgb_results.values()]),
            'Accuracy': np.mean([r['accuracy'] for r in xgb_results.values()])
        }
        
        # Create radar chart comparison
        metrics = list(rf_metrics.keys())
        rf_values = list(rf_metrics.values())
        xgb_values = list(xgb_metrics.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, rf_values, width, label='Random Forest', alpha=0.8, color='#FF6B6B')
        bars2 = ax1.bar(x + width/2, xgb_values, width, label='XGBoost', alpha=0.8, color='#4ECDC4')
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Average Performance Comparison: Random Forest vs XGBoost')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # Training time comparison
        rf_times = [r['training_time'] for r in rf_results.values()]
        xgb_times = [r['training_time'] for r in xgb_results.values()]
        
        ax2.boxplot([rf_times, xgb_times], labels=['Random Forest', 'XGBoost'])
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_best_models_summary(self):
        """Show summary of best performing models"""
        print("\n" + "="*100)
        print("BEST MODELS SUMMARY")
        print("="*100)
        
        # Best by different criteria
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])
        best_roc_auc = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        
        criteria = [
            ('F1-Score', best_f1),
            ('Precision', best_precision),
            ('Recall', best_recall),
            ('ROC-AUC', best_roc_auc),
            ('Accuracy', best_accuracy)
        ]
        
        for criterion, (model_key, result) in criteria:
            print(f"\nBest {criterion}: {result['model_name']} ({result['sampling_method']})")
            print(f"  Algorithm: {result['algorithm']}")
            print(f"  F1: {result['f1_score']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}")
            print(f"  Accuracy: {result['accuracy']:.4f}, ROC-AUC: {result['roc_auc']:.4f}")
    
    def enhanced_analysis(self):
        """Perform enhanced analysis including cross-validation and feature importance"""
        print("\n" + "="*80)
        print("ENHANCED ANALYSIS")
        print("="*80)
        
        # Cross-validation analysis
        self.perform_cross_validation_analysis()
        
        # Feature importance analysis
        self.enhanced_feature_importance_analysis()
        
        # Statistical significance testing
        self.statistical_significance_testing()
        
    def perform_cross_validation_analysis(self):
        """Perform cross-validation analysis on top models"""
        print("\n" + "="*50)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*50)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {}
        
        # Get top 6 performing models
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:6]
        
        for i, (model_key, result) in enumerate(top_models):
            print(f"\nPerforming CV on {result['model_name']} ({result['sampling_method']})...")
            
            # Get the trained model and scaler
            model = result['model']
            
            # Find the corresponding scaler from self.models
            scaler = None
            for mk, mi in self.models.items():
                if mk == model_key:
                    scaler = mi['scaler']
                    break
            
            if scaler is None:
                print(f"  Scaler not found for {model_key}, skipping...")
                continue
            
            # Scale the training data
            X_train_scaled = scaler.transform(self.X_train)
            
            try:
                # Perform cross-validation
                cv_scores = {
                    'f1': cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='f1'),
                    'precision': cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='precision'),
                    'recall': cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='recall'),
                    'roc_auc': cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
                }
                
                cv_results[model_key] = {
                    'model_name': result['model_name'],
                    'sampling': result['sampling_method'],
                    'algorithm': result['algorithm'],
                    'cv_scores': cv_scores,
                    'test_f1': result['f1_score']
                }
                
                # Print CV results
                print(f"  CV F1: {cv_scores['f1'].mean():.4f} ± {cv_scores['f1'].std():.4f}")
                print(f"  CV ROC-AUC: {cv_scores['roc_auc'].mean():.4f} ± {cv_scores['roc_auc'].std():.4f}")
                print(f"  Test F1 (original): {result['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  CV failed for {model_key}: {e}")
                continue
        
        return cv_results
    
    def enhanced_feature_importance_analysis(self):
        """Enhanced feature importance analysis"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get the best RF and XGB models
        best_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        best_rf = None
        best_xgb = None
        
        for model_key, result in best_models:
            if result['algorithm'] == 'Random Forest' and best_rf is None:
                best_rf = (model_key, result)
            elif result['algorithm'] == 'XGBoost' and best_xgb is None:
                best_xgb = (model_key, result)
            
            if best_rf and best_xgb:
                break
        
        if best_rf and best_xgb:
            rf_model = best_rf[1]['model']
            xgb_model = best_xgb[1]['model']
            
            # Get feature importance
            rf_importance = rf_model.feature_importances_
            xgb_importance = xgb_model.feature_importances_
            
            # Create feature importance dataframe
            n_features = len(rf_importance)
            feature_names = self.feature_names[:n_features] if self.feature_names else [f'Feature_{i}' for i in range(n_features)]
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'RF_Importance': rf_importance,
                'XGB_Importance': xgb_importance
            })
            
            # Calculate correlation between importance rankings
            importance_corr = np.corrcoef(rf_importance, xgb_importance)[0, 1]
            
            print(f"Feature Importance Correlation: {importance_corr:.4f}")
            
            # Top 10 features for each algorithm
            rf_top = feature_importance_df.nlargest(10, 'RF_Importance')
            xgb_top = feature_importance_df.nlargest(10, 'XGB_Importance')
            
            print(f"\nTop 10 Features - Random Forest ({best_rf[1]['model_name']}):")
            for idx, row in rf_top.iterrows():
                print(f"  {row['Feature']}: {row['RF_Importance']:.4f}")
            
            print(f"\nTop 10 Features - XGBoost ({best_xgb[1]['model_name']}):")
            for idx, row in xgb_top.iterrows():
                print(f"  {row['Feature']}: {row['XGB_Importance']:.4f}")
            
            # Plot comparison
            try:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
                
                # RF feature importance
                ax1.barh(range(len(rf_top)), rf_top['RF_Importance'], color='#FF6B6B', alpha=0.8)
                ax1.set_yticks(range(len(rf_top)))
                ax1.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in rf_top['Feature']])
                ax1.set_title('Top 10 Features - Random Forest')
                ax1.set_xlabel('Importance')
                
                # XGB feature importance
                ax2.barh(range(len(xgb_top)), xgb_top['XGB_Importance'], color='#4ECDC4', alpha=0.8)
                ax2.set_yticks(range(len(xgb_top)))
                ax2.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in xgb_top['Feature']])
                ax2.set_title('Top 10 Features - XGBoost')
                ax2.set_xlabel('Importance')
                
                # Correlation scatter plot
                ax3.scatter(feature_importance_df['RF_Importance'], feature_importance_df['XGB_Importance'], alpha=0.6)
                ax3.set_xlabel('Random Forest Importance')
                ax3.set_ylabel('XGBoost Importance')
                ax3.set_title(f'Feature Importance Correlation\n(r = {importance_corr:.3f})')
                ax3.grid(True, alpha=0.3)
                
                # Add diagonal line
                max_imp = max(rf_importance.max(), xgb_importance.max())
                ax3.plot([0, max_imp], [0, max_imp], 'r--', alpha=0.5)
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error creating feature importance plots: {e}")
            
            return feature_importance_df
        else:
            print("Could not find suitable RF and XGB models for comparison.")
            return None
    
    def statistical_significance_testing(self):
        """Statistical significance testing between algorithms"""
        print("\n" + "="*50)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*50)
        
        # Separate results by algorithm
        rf_results = [r for r in self.results.values() if r['algorithm'] == 'Random Forest']
        xgb_results = [r for r in self.results.values() if r['algorithm'] == 'XGBoost']
        
        if len(rf_results) > 0 and len(xgb_results) > 0:
            # Get F1 scores
            rf_f1_scores = [r['f1_score'] for r in rf_results]
            xgb_f1_scores = [r['f1_score'] for r in xgb_results]
            
            # Get AUC scores
            rf_auc_scores = [r['roc_auc'] for r in rf_results]
            xgb_auc_scores = [r['roc_auc'] for r in xgb_results]
            
            # Perform statistical tests
            try:
                t_stat_f1, t_pvalue_f1 = stats.ttest_ind(rf_f1_scores, xgb_f1_scores)
                t_stat_auc, t_pvalue_auc = stats.ttest_ind(rf_auc_scores, xgb_auc_scores)
                
                print(f"\nF1 Score Comparison:")
                print(f"Random Forest - Mean: {np.mean(rf_f1_scores):.4f}, Std: {np.std(rf_f1_scores):.4f}")
                print(f"XGBoost - Mean: {np.mean(xgb_f1_scores):.4f}, Std: {np.std(xgb_f1_scores):.4f}")
                print(f"T-test: t-statistic = {t_stat_f1:.4f}, p-value = {t_pvalue_f1:.4f}")
                
                print(f"\nROC-AUC Comparison:")
                print(f"Random Forest - Mean: {np.mean(rf_auc_scores):.4f}, Std: {np.std(rf_auc_scores):.4f}")
                print(f"XGBoost - Mean: {np.mean(xgb_auc_scores):.4f}, Std: {np.std(xgb_auc_scores):.4f}")
                print(f"T-test: t-statistic = {t_stat_auc:.4f}, p-value = {t_pvalue_auc:.4f}")
                
                alpha = 0.05
                if t_pvalue_f1 < alpha:
                    print(f"\n✓ Statistically significant difference in F1 scores (p < {alpha})")
                    if np.mean(rf_f1_scores) > np.mean(xgb_f1_scores):
                        print("  → Random Forest performs significantly better in F1")
                    else:
                        print("  → XGBoost performs significantly better in F1")
                else:
                    print(f"\n✗ No statistically significant difference in F1 scores (p >= {alpha})")
                
            except Exception as e:
                print(f"Error in statistical testing: {e}")
        else:
            print("Need results from both algorithms for statistical testing.")
    
    def run_comprehensive_analysis(self):
        """Run the complete enhanced analysis"""
        print("ENHANCED RANDOM FOREST vs XGBOOST COMPARISON WITH CONFUSION MATRICES")
        print("=" * 80)
        
        # Step 1: Load data
        if not self.load_data():
            print("Failed to load data. Please check the setup.")
            return False
        
        # Step 2: Advanced preprocessing
        self.advanced_preprocessing()
        
        # Step 3: Split data
        self.split_data()
        
        # Step 4: Prepare datasets with different sampling techniques
        self.prepare_datasets_with_sampling()
        
        # Step 5: Train Random Forest variants
        self.train_random_forest_variants()
       
        # Step 6: Train XGBoost variants
        self.train_xgboost_variants()
        
        # Step 7: Evaluate all models
        self.evaluate_all_models()
        
        # Step 8: Create comprehensive comparison table
        comparison_df = self.create_comprehensive_comparison_table()
        
        # Step 9: Show confusion matrices and ROC curves
        self.plot_roc_curves_comparison()
        self.plot_top_confusion_matrices(top_n=8)
        self.plot_all_confusion_matrices()
        
        # Step 10: Confusion matrix analysis 
        self.create_confusion_matrix_summary()
        self.plot_performance_vs_confusion_metrics()
        
        # Step 11: Visualizations and analysis
        self.plot_algorithm_performance_comparison()
        
        # Step 12: Show best models summary
        self.show_best_models_summary()
        
        # Step 13: Enhanced analysis
        self.enhanced_analysis()
        
        print("\n" + "="*80)
        print("ENHANCED ANALYSIS WITH CONFUSION MATRICES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return True

def main():
    """Main function to run the analysis"""
    analyzer = EnhancedRF_XGB_Comparison()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        return analyzer
    else:
        return None

# Run the enhanced analysis
if __name__ == "__main__":
    analyzer = main()
