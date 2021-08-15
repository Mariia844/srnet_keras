import os, csv


with open('history.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader, None)
    losses = []
    aucs = []
    for row in reader:
        arr = []
        for value in row:
            arr.append(value)
        if len(arr) > 0:
            loss, val_loss, auc, val_auc = arr[1], arr[4], arr[3], arr[6]
            losses.append((loss, val_loss))
            aucs.append((auc, val_auc))
            # print(f'Loss: {loss}, val_loss: {val_loss}, auc: {auc}, val_auc: {val_auc}')
    # for auc in aucs:
    for auc in aucs:
        

    print(f'Max AUC: {max_auc}, index is {index}') 