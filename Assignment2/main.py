from model import LeNet_model
from data import MyDataset, prepare_dataset

if __name__ == "__main__":
    model = LeNet_model()
    print("Model built.")
    train, test, train_split = prepare_dataset()
    print("Data loaded.")

    print("Original train set size:", len(train))
    print("Split train set size:", len(train_split))
    print("Test set size:", len(test))

    print("Training on original train set...") 
    model.fit(train,
        epochs=5,
        batch_size=64,
        verbose=1
    )
    model.evaluate(test, batch_size=64, verbose=1)
    model.save('ckpt/origin')

    print("Training on split train set...")
    model = LeNet_model()
    model.fit(MyDataset(train_split),
        epochs=5,
        batch_size=64,
        verbose=1
    )
    model.evaluate(test, batch_size=64, verbose=1)
    model.save('ckpt/split')

