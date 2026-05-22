import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Set premium styling constants to match the web app dark theme
THEME_BG = '#070913'       # Deep space navy background
CARD_BG = '#0f1123'        # Frosted glass card background
TEXT_COLOR = '#e2e8f0'     # Slate-200 text
GRID_COLOR = '#1e293b'     # Slate-800 grid lines
COLOR_SETOSA = '#38bdf8'    # Cyan-400
COLOR_VERSICOLOR = '#f43f5e' # Rose-500
COLOR_VIRGINICA = '#a855f7'  # Purple-500
ACCENT_COLORS = [COLOR_SETOSA, COLOR_VERSICOLOR, COLOR_VIRGINICA]
FONT_FAMILY = 'sans-serif'

# Configure Matplotlib styling
plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.facecolor': THEME_BG,
    'figure.facecolor': THEME_BG,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.5,
    'savefig.facecolor': THEME_BG,
    'savefig.edgecolor': THEME_BG
})

def train_and_export():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = [name.replace(" (cm)", "").replace(" ", "_") for name in iris.feature_names]
    target_names = list(iris.target_names)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    metrics_summary = {}
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    # We will store trained models for later use
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        metrics_summary[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
        
        print(f"{name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    # Ensure output directories exist
    os.makedirs('assets', exist_ok=True)

    # ==================== VISUALIZATION 1: CONFUSION MATRICES ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(THEME_BG)
    
    for idx, (name, model) in enumerate(trained_models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Create a beautiful custom heatmap with custom colors
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax, cbar=False,
                    xticklabels=target_names, yticklabels=target_names,
                    annot_kws={"size": 16, "weight": "bold", "color": "#ffffff"},
                    linewidths=1.5, linecolor=THEME_BG)
        
        # Style heatmap cells manually to look premium
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=16, pad=15, weight='bold', color=TEXT_COLOR)
        ax.set_xlabel("Predicted Species", fontsize=12, labelpad=10, color=TEXT_COLOR)
        ax.set_ylabel("True Species", fontsize=12, labelpad=10, color=TEXT_COLOR)
        ax.set_xticklabels(target_names, fontsize=11)
        ax.set_yticklabels(target_names, fontsize=11, rotation=0)
        
        # Border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(1.5)
            
    plt.tight_layout()
    plt.savefig('assets/confusion_matrices.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print("Confusion matrices visualization saved.")

    # ==================== VISUALIZATION 2: ROC CURVES ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(THEME_BG)
    
    for idx, (name, model) in enumerate(trained_models.items()):
        ax = axes[idx]
        y_prob = model.predict_proba(X_test)
        
        # Plot ROC curve for each class One-vs-Rest
        for class_idx in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=ACCENT_COLORS[class_idx], lw=2.5,
                    label=f'{target_names[class_idx].capitalize()} (AUC = {roc_auc:.2f})')
            
        ax.plot([0, 1], [0, 1], color='#475569', linestyle='--', lw=1.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12, labelpad=10)
        ax.set_ylabel('True Positive Rate', fontsize=12, labelpad=10)
        ax.set_title(f'{name}\nROC Curves', fontsize=16, pad=15, weight='bold')
        ax.legend(loc="lower right", framealpha=0.15, facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Border
        for _, spine in ax.spines.items():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(1.5)
            
    plt.tight_layout()
    plt.savefig('assets/roc_curves.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print("ROC Curves saved.")

    # ==================== VISUALIZATION 3: FEATURE IMPORTANCE ====================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(THEME_BG)
    
    tree_models = {
        'Decision Tree': trained_models['Decision Tree'],
        'Random Forest': trained_models['Random Forest']
    }
    
    clean_feature_names = [f.replace('_', ' ').title() for f in feature_names]
    
    for idx, (name, model) in enumerate(tree_models.items()):
        ax = axes[idx]
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        # Dynamic colored bars
        colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(importances))]
        # Sort colors to align with importances
        colors = [colors[i] for i in indices]
        
        ax.barh(range(len(importances)), importances[indices], color=colors, height=0.6, align='center', alpha=0.95)
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([clean_feature_names[i] for i in indices], fontsize=11, weight='bold')
        ax.set_xlabel('Gini Importance', fontsize=12, labelpad=10)
        ax.set_title(f'{name}\nFeature Importance', fontsize=15, pad=15, weight='bold')
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)
        
        # Add values next to bars
        for i, v in enumerate(importances[indices]):
            if v > 0.01:
                ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10, weight='bold', color=TEXT_COLOR)
                
        # Border
        for _, spine in ax.spines.items():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(1.5)
            
    plt.tight_layout()
    plt.savefig('assets/feature_importance.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print("Feature importance saved.")

    # ==================== VISUALIZATION 4: DECISION BOUNDARIES ====================
    # We will project the model decision boundaries using Petal Length and Petal Width (the 2 most important features)
    # To do this cleanly, we train secondary models specifically on these 2 features just for this visual representation.
    # This allows a clear 2D plot.
    X_2d = X[:, [2, 3]] # Petal Length, Petal Width
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)
    
    models_2d = {
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.patch.set_facecolor(THEME_BG)
    
    # Create grid for boundaries
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Custom colormap for filled regions: cyan, rose, purple with low alpha
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#0e2d42', '#3f1122', '#28113c']) # Subdued background colors
    cmap_bold = ListedColormap(ACCENT_COLORS)
    
    for idx, (name, model) in enumerate(models_2d.items()):
        ax = axes[idx]
        model.fit(X_train_2d, y_train_2d)
        
        # Predict on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot filled contours
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.9)
        
        # Plot decision boundary lines
        ax.contour(xx, yy, Z, colors=GRID_COLOR, linewidths=0.8, alpha=0.5)
        
        # Scatter actual data points
        for class_idx, color in enumerate(ACCENT_COLORS):
            points = X_2d[y == class_idx]
            ax.scatter(points[:, 0], points[:, 1], color=color, edgecolor='#070913',
                       label=target_names[class_idx].capitalize(), s=40, alpha=0.85, linewidth=0.5)
            
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Petal Length (cm)', fontsize=12, labelpad=10)
        ax.set_ylabel('Petal Width (cm)', fontsize=12, labelpad=10)
        ax.set_title(f'{name}\nDecision Boundary (Petals)', fontsize=15, pad=15, weight='bold')
        ax.grid(True, linestyle=':', alpha=0.3)
        
        if idx == 2:
            ax.legend(loc="lower right", framealpha=0.15, facecolor=CARD_BG, edgecolor=GRID_COLOR, fontsize=10)
            
        # Border
        for _, spine in ax.spines.items():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(1.5)
            
    plt.tight_layout()
    plt.savefig('assets/decision_boundaries.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print("Decision boundaries saved.")

    # ==================== TRANSPILE DECISION TREE TO JAVASCRIPT ====================
    dt_model = trained_models['Decision Tree']
    
    def export_tree_to_js(tree_clf, feature_names, class_names):
        tree = tree_clf.tree_
        
        def recurse(node, depth):
            indent = "  " * depth
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Build tree logic recursively
                js_code = f"{indent}if (features.{name} <= {threshold:.4f}) {{\n"
                js_code += recurse(tree.children_left[node], depth + 1)
                js_code += f"{indent}}} else {{\n"
                js_code += recurse(tree.children_right[node], depth + 1)
                js_code += f"{indent}}}\n"
                return js_code
            else:
                # Leaf node: get class probabilities
                value = tree.value[node][0]
                probs = value / np.sum(value)
                predicted_class = class_names[np.argmax(probs)]
                
                js_code = f"{indent}return {{\n"
                js_code += f"{indent}  prediction: '{predicted_class}',\n"
                js_code += f"{indent}  probabilities: {list(probs.round(4))},\n"
                js_code += f"{indent}  path: 'Leaf Node reached'\n"
                js_code += f"{indent}}};\n"
                return js_code
                
        return recurse(0, 1)

    # Let's write a visual tracer path in JS!
    # A cleaner tracer return that outputs path step-by-step
    def export_tree_to_js_with_path(tree_clf, feature_names, class_names):
        tree = tree_clf.tree_
        
        def recurse(node, path_str):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                clean_name = name.replace('_', ' ').title()
                threshold = tree.threshold[node]
                
                left_path = f"{path_str} -> {clean_name} &le; {threshold:.2f}cm"
                right_path = f"{path_str} -> {clean_name} > {threshold:.2f}cm"
                
                js_code = f"if (features.{name} <= {threshold:.4f}) {{\n"
                js_code += recurse(tree.children_left[node], left_path)
                js_code += "} else {\n"
                js_code += recurse(tree.children_right[node], right_path)
                js_code += "}\n"
                return js_code
            else:
                value = tree.value[node][0]
                probs = value / np.sum(value)
                predicted_class = class_names[np.argmax(probs)]
                
                js_code = f"return {{\n"
                js_code += f"  prediction: '{predicted_class}',\n"
                js_code += f"  probabilities: {list(probs.round(4))},\n"
                js_code += f"  path: '{path_str}'\n"
                js_code += f"}};\n"
                return js_code
                
        return recurse(0, "Root")

    js_tree_code = export_tree_to_js_with_path(dt_model, feature_names, target_names)
    
    # Export full dataset for the interactive scatter plot
    data_list = []
    for i in range(len(X)):
        data_list.append({
            'sepal_length': float(X[i, 0]),
            'sepal_width': float(X[i, 1]),
            'petal_length': float(X[i, 2]),
            'petal_width': float(X[i, 3]),
            'species': target_names[y[i]]
        })
        
    # Calculate dataset statistics for comparison (means of each feature per species)
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[val] for val in y]
    means = df.groupby('species').mean().to_dict(orient='index')
    
    # Create final model_data.js content
    js_content = f"""// Auto-generated configuration and model details
const MODEL_METRICS = {json.dumps(metrics_summary, indent=2)};

const IRIS_DATASET = {json.dumps(data_list, indent=2)};

const SPECIES_MEANS = {json.dumps(means, indent=2)};

// 100% faithful client-side Decision Tree classifier compiled from Python sklearn
function predictDecisionTree(features) {{
  // features keys: sepal_length, sepal_width, petal_length, petal_width
  {js_tree_code}
}}
"""
    
    with open('model_data.js', 'w') as f:
        f.write(js_content)
        
    print("JS model config successfully exported to model_data.js")

if __name__ == '__main__':
    train_and_export()
