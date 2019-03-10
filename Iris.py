import math
import matplotlib.pyplot as plt

def delta(x, _type, sigmoid):
    return (2 * x * (sigmoid-_type) * (1-sigmoid) * sigmoid)

def lossFunction(_type, sigmoid):
    return ((_type-sigmoid)**2)/2

def sigmoidFunction(result):
    return 1 / (1 + math.exp(-result))

def activationFunction(sigmoid):
    if(sigmoid > 0.5):
        return 1
    else:
        return 0

def training(dataTraining, theta, bias, alpha):
    ret = {}
    loss = 0
    accuracy = 0
    for data in dataTraining:
        result = 0 # hitung result
        for i in range(len(theta)):
            result += data['input'][i]*theta[i]
        result += bias

        sigmoid = sigmoidFunction(result)
        for i in range(len(theta)): # update nilai theta dan bias
            theta[i] -= alpha*delta(data['input'][i], data['type'], sigmoid)
        bias -= alpha*delta(1, data['type'], sigmoid)

        loss += lossFunction(data['type'], sigmoid) # hitung total loss dan accuracy
        if(data['type'] == activationFunction(sigmoid)):
            accuracy += 1
            
    ret['bias'] = bias
    ret['accuracy'] = accuracy/float(len(dataTraining)) # rata-rata dari semua data
    ret['loss'] = loss/float(len(dataTraining))
    return ret

def validation(dataValidation, theta, bias):
    ret = {}
    loss = 0
    accuracy = 0
    for data in dataValidation:
        result = 0 # hitug result
        for i in range(len(theta)):
            result += data['input'][i]*theta[i]
        result += bias

        sigmoid = sigmoidFunction(result)
        loss += lossFunction(data['type'], sigmoid) # hitung total loss dan accuracy
        if(data['type'] == activationFunction(sigmoid)):
            accuracy += 1

    ret['accuracy'] = accuracy/float(len(dataValidation)) #rata-rata dari semua data
    ret['loss'] = loss/float(len(dataValidation))
    return ret

def SLP(theta, bias, alpha, data, epoch):
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    for _ in range(epoch):
        train_acc = 0
        train_los = 0
        valid_acc = 0
        valid_los = 0

        for k in range(len(data)): 
            dataTraining = [] # data untuk training
            dataValidation = [] # data untuk validasi
            for i in range(len(data)):
                if(i == k):
                    dataValidation += data[i]
                else:
                    dataTraining += data[i]
            # rata-rata accuracy dan loss dari tiap iterasi dijumlah
            temp = training(dataTraining, theta, bias, alpha)
            bias = temp['bias']
            train_acc += temp['accuracy']
            train_los += temp['loss']
            temp = validation(dataValidation, theta, bias)
            valid_acc += temp['accuracy']
            valid_los += temp['loss']
        
        training_accuracy.append(train_acc/float(len(data))) # jumlah accuracy dan loss per epoch dirata-rata
        training_loss.append(train_los/float(len(data)))
        validation_accuracy.append(valid_acc/float(len(data)))
        validation_loss.append(valid_los/float(len(data)))

    # menggambar grafik
    plt.figure(1)
    plt.subplot(211)
    plt.plot(training_loss, label='Training')
    plt.plot(validation_loss, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title('Loss Function, alpha = {}'.format(alpha), loc='left')
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(training_accuracy, label='Training')
    plt.plot(validation_accuracy, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Accuracy')
    plt.title('Accuracy, alpha = {}'.format(alpha), loc='left')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    f = open("Iris.csv")
    dataRaw = f.read()
    f.close()

    dataRaw = [[x for x in y.split(',')] for y in dataRaw.split()] # data dijadikan array 2D
    data = []

    for n in dataRaw:
        temp = {}
        temp['input'] = [float(n[0]), float(n[1]), float(n[2]), float(n[3])] # input (x)
        temp['name'] = n[4] # nama
        if(n[4] == 'Iris-setosa'): # type 0/1
            temp['type'] = 0
        else:
            temp['type'] = 1
        data.append(temp)

    data = [data[x:x+20] for x in range(0, len(data), 20)] # data dibagi menjadi 5 kelompok
    theta = [0.3, 0.1, 0.8, 0.9]
    SLP(theta, 0.5, 0.1, data, 300)
    theta = [0.3, 0.1, 0.8, 0.9]
    SLP(theta, 0.5, 0.8, data, 300)

if __name__=="__main__":
    main()          
