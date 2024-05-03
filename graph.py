#draw a graph base on the data in out.txt
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    accuracy = []
    loss = []
    
    with open("out.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            each = line.split()
            loss.append(float(each[2]))
            accuracy.append(float(each[4]))

    print(accuracy)
    print(loss)

    #plot a graph with accuracy
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.show()
    
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.show()
    