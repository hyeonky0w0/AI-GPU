def fit_and_predict(train_x, train_y, valid_x, config):
    """Small dependency-free baseline; replace with MLP or CatBoost per branch."""
    learning_rate = config["model"].get("learning_rate", 0.1)
    epochs = config["model"].get("epochs", 200)
    weights = [0.0] * len(train_x[0])
    bias = 0.0

    for _ in range(epochs):
        predictions = [sum(w * x for w, x in zip(weights, row)) + bias for row in train_x]
        errors = [prediction - target for prediction, target in zip(predictions, train_y)]
        scale = 2.0 / len(train_x)
        for column in range(len(weights)):
            gradient = scale * sum(error * row[column] for error, row in zip(errors, train_x))
            weights[column] -= learning_rate * gradient
        bias -= learning_rate * scale * sum(errors)

    return [sum(w * x for w, x in zip(weights, row)) + bias for row in valid_x]
