ufrom tqdm.auto import tqdm 
import torch
import torch.nn.functional as F

def train(args, model, train_loader, valid_loader, optimizer, criterion):
    model.train()
    for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
        if epoch == args.freeze_resnet:
            model.unfreeze()
        
        for batch in train_loader:
            xs = [b.to(args.device) for b in batch]
            
            xc = torch.cat(xs, axis=0)
            
            embeds, logits = model(xc)
            
            optimizer.zero_grad()
            
            y = torch.arange(len(xs), device=args.device)
            y = y.repeat_interleave(xs[0].size(0))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            scheduler.step(epoch)
            
            pred_y = torch.argmax(logits, axis=1)
            
            accuracy = torch.sum(pred_y == y).detach().cpu().item()
            
            roc_auc = evaluate(args, model, valid_loader, criterion)
        
        torch.save(model.state_dict(), f'model_parameters/models.pt')
        
        
def evaluate(args, model, valid_loader, criterion):
    labels, embeds = [], []
    
    with torch.no_grad():
        for batch in valid_loader:
            batch = [b.to(args.device) for b in batch]
            
            embed, logit = model(x)
            
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    
    labels = torch.cat(labels)
    embeds = torch.cat(embeds)
    
    if train_embed is None:
        train_embed = get_train_embeds(model, size, defect_type, test_transform, device)
        
    embeds = F.normalize(embeds, p=2, dim=1)
    train_embed = F.normalize(train_embed, p=2, dim=1)
    
    
    density.fit(train_embed)
    distances = density.predict(embeds)
    
    roc_auc = auc_measure(labels, distances)
    
    return roc_auc
