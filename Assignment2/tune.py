from main import LeNet_model, MyDataset, prepare_dataset



if __name__ == "__main__":
    train, test, train_split = prepare_dataset()
    model = LeNet_model(1e-2, detach_feats=True)
    model.load("ckpt/split", reset_optimizer=True)

    test_x = [x for x, y in test]
    self_legs = model.predict(MyDataset(test_x), 64, 1, True)[0]
    self_labels = self_legs.argmax(-1).reshape(-1, 1)
    model.fit(MyDataset(list(zip(test_x, self_labels)) + train_split),
        epochs=1,
        batch_size=64,
        verbose=1
    )
    model.evaluate(test, batch_size=64, verbose=1)
    model.save('ckpt/split_tune')