def SE(predicted, target, derivate=False):
    # Squared Error
    if not derivate:
        return pow(predicted-target, 2)
    else:
        return 2(predicted-target)