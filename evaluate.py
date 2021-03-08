# Evaluation Function
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def evaluate(model, test_loader,device, des_folder, threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for ((text, text_len), labels), _ in test_loader:      
                labels = labels.to(device)
                text = text.to(device)
                text_len = text_len.to(device)
                output = model(text, text_len)
                output = (output > threshold).int()
                y_pred.extend(output.tolist())
                y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['POSITIVE', 'NEGATIVE'])
    ax.yaxis.set_ticklabels(['POSITIVE', 'NEGATIVE'])
    plt.savefig(des_folder + '/eval.png')