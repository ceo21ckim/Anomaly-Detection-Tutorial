import dgl.function as fn 
import sympy
import scipy

import torch 
import numpy as np 

import torch.nn.functional as F
import dgl.function as fn 

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix

from settings import * 


def unnLaplacian(feat, D_invsqrt, graph):
    graph.ndata['h'] = feat * D_invsqrt 
    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    return feat - graph.ndata.pop('h') * D_invsqrt 


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / scipy.special.beta(i+1, d+1-i))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas 


def get_mask(args, g):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))

    if args.d_name == 'amazon':
        index = list(range(3305, len(labels)))
    
    idx_train, idx_rest, y_train, y_rest = train_test_split(
        index, labels[index], stratify=labels[index], 
        train_size=args.train_ratio, random_state=42, shuffle=True)
    
    idx_valid, idx_test, y_valid, y_test = train_test_split(
        idx_rest, y_rest, stratify=y_rest, 
        test_size=0.67, random_state=42, shuffle=True)
    
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()
    
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1 
    
    return features, labels, train_mask, val_mask, test_mask 


def train(args, model, graph, optimizer):
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    
    features, labels, train_mask, val_mask, test_mask = get_mask(args, graph)
    
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight:', weight)
    
    model.train()
    for epoch in range(args.num_epochs):
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        f1, trec, tpre, tmf1, tauc = evaluate(model, labels, logits, val_mask, test_mask)
        
        if best_f1 < f1:
            best_f1 = f1 
            
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc 
            
            save_path = os.path.join(BASE_DIR, 'model.pt')
            torch.save(model.state_dict(), save_path)
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch: [{epoch+1}/{args.num_epochs}], loss: {loss:.4f}, val_mf1: {best_f1:.4f}')

    print(f'Test: Recall {final_trec*100:.2f}, Precision {final_tpre*100:.2f}, Macro f1 {final_tmf1*100:.2f}, Accuracy {final_tauc*100:.2f}')
    
    return final_tmf1, final_tauc


def evaluate(model, labels, logits, val_mask, test_mask):

    model.eval()
    probs = logits.softmax(1)
    f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
    preds = np.zeros_like(labels)
    preds[probs[:, 1] > thres] = 1 
    
    trec = recall_score(labels[test_mask], preds[test_mask])
    tpre = precision_score(labels[test_mask], preds[test_mask])
    tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
    tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
    return f1, trec, tpre, tmf1, tauc 


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1 
            best_thre = thres 
    return best_f1, best_thre 
