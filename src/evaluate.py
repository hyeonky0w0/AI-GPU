import math


def evaluate(actual, predicted):
    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    return {
        "val_rmse": round(math.sqrt(sum(error * error for error in errors) / len(errors)), 6),
        "val_mae": round(sum(abs(error) for error in errors) / len(errors), 6),
    }
