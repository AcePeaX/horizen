import matplotlib.pyplot as plt
import csv
import os

toPlot = ["pre-horizen-2BT-87M-maserati.zen.csv",
          "pre-horizen-2BT-204M.zen.csv",
          "pre-horizen-4BT-1024VS-87M.zen.csv",
          "pre-horizen-4BT-1024VS-204M.zen.csv"]

if type(toPlot)==str:
    toPlot = [toPlot]
for fineRelativePath in toPlot:
    csvfile_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../logs',fineRelativePath))
    filename = '.'.join(os.path.basename(csvfile_path).split('.')[:-1])
    dirname = os.path.dirname(csvfile_path)

    with open(csvfile_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        X = []
        X_val = []
        lr = []
        train_loss = []
        val_loss = []
        first = True
        for row in reader:
            if first:
                first=False
                continue
            X.append(int(row[0]))
            train_loss.append(float(row[1]))
            lr.append(float(row[3]))
            if row[2]!='':
                X_val.append(int(row[0]))
                val_loss.append(float(row[2]))

    w, h = 11, 20
    fig = plt.figure(figsize=(w, h))
    ax1, ax2, ax3 = fig.subplots(nrows=3)

    ax1.plot(X, train_loss)
    ax1.set_title('Train loss')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss')

    ax2.plot(X, lr)
    ax2.ticklabel_format(axis='y',style='sci', scilimits=(0, 0))
    ax2.set_title('Learning rate')
    ax2.set_xlabel('steps')
    ax2.set_ylabel('rate')

    ax3.plot(X_val, val_loss)
    ax3.set_title('Validation loss')
    ax3.set_xlabel('steps')
    ax3.set_ylabel('loss')


    output_path = os.path.join(dirname, filename+".png")

    plt.savefig(output_path)