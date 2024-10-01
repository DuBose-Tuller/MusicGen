from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import os

EMBEDDING_SIZE = 1536

def process_json(file):
    with open(file, "r") as f:
        embeddings_dict = json.load(f)

    embeddings = np.array(list(embeddings_dict.values()))
    return embeddings

def get_filenames(sources):
    files = []
    for data in sources:
        sampling = f"s{data['segment']}-t{data['stride']}"
        path = os.path.join(data['dataset'], sampling)
        if os.path.exists(path):
            filename = os.path.join(path, f"{data['method']}_embeddings.json")
            files.append(filename)       

    if files == []:
        raise ValueError
    
    return files
    
def construct_dataset(sources, verbose=False):
    X_arrays = []
    y_arrays = []
    for (i, source) in enumerate(sources):
        embeddings = process_json(source)
        labels = np.full((embeddings.shape[0],), i)
        X_arrays.append(embeddings)
        y_arrays.append(labels)

    X = np.concatenate(X_arrays, axis=0)
    y = np.concatenate(y_arrays, axis=None)

    return X, y

def multiclass_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Increase max_iter and add scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,  # Increase from default 100
        tol=1e-4        # Optionally, adjust tolerance
    ).fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Additional metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "roc-auc": roc_auc,
    }
    
    return matrix, metrics



def main():
    config = [
        {
            "dataset": "acpas",
            "method": "last",
            "segment": "30",
            "stride": "30"
        },
        {
            "dataset": "CBF",
            "method": "last",
            "segment": "30",
            "stride": "30"
        }
    ]
    
    files = get_filenames(config)
    X, y = construct_dataset(files)

    cm, metrics = multiclass_model(X, y)
    print(type(metrics))
    print(cm)
    print(f"F1 Score: {metrics['f1']:0.2f}")
    print(f"Precision: {metrics['precision']:0.2f}")
    print(f"Recall: {metrics['recall']:0.2f}")
    print(f"ROC-AUC: {metrics['roc-auc']:0.2f}")

if __name__ == "__main__":
    main()