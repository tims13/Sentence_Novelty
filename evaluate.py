# Evaluation Function
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns

def evaluate(model, train_loader, test_loader, novel_loader, device, des_folder, threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    train_features = []
    novel_features = []
    # test_features = []
    with torch.no_grad():
        # on test set, predict
        for ((text, text_len), labels), _ in test_loader:      
                labels = labels.to(device)
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)
                # test_features.append(model.deep_features.cpu().numpy())
                output = (output > threshold).int()
                y_pred.extend(output.tolist())
                y_true.extend(labels.tolist())

        # on train set, obtain deep features
        for ((text, text_len), labels), _ in train_loader:      
                labels = labels.to(device)
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)
                train_features.append(model.deep_features.cpu().numpy())

        # on novel set, obtain deep features
        for ((text, text_len), labels), _ in novel_loader:      
                labels = labels.to(device)
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)
                novel_features.append(model.deep_features.cpu().numpy())

    # test_features = np.vstack(test_features)
    train_features = np.vstack(train_features)
    novel_features = np.vstack(novel_features)
    # compute the prediction
    y_pred = np.array(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.array(y_true)
    print('y_pred:')
    print(y_pred)
    print('y_true:')
    print(y_true)
    # The report of the classification task
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    # The confusion_matrix of the classification task
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['POSITIVE', 'NEGATIVE'])
    ax.yaxis.set_ticklabels(['POSITIVE', 'NEGATIVE'])
    plt.savefig(des_folder + '/eval.png')

    # LOF
    print('LOF training...')
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.5, novelty=True, n_jobs=-1)
    lof.fit(train_features)
    y_pred_lof = lof.predict(novel_features)
    lof_recall = sum(y_pred_lof == -1) / len(y_pred_lof)
    print('LOF RECALL:')
    print(lof_recall)
    