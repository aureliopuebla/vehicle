import random
import numpy as np

# To be used only with this parameters for the presentation.
# Afterwards, this file will be deleted.
RANSAC_TRIES = 2000
RANSAC_EPSILON = 1
MAX_DISPARITY = 16383
HISTOGRAM_BINS = 256
BIN_SIZE = (MAX_DISPARITY + 1) / HISTOGRAM_BINS
bestM = 0.0
bestB = 0.0


def evaluateRANSACTry(VdispImage, m, b):
    """Tests how fit a certain RANSAC try is in fitting the road plane."""
    rows, cols = VdispImage.shape
    f = 0
    for x in range(cols):
        y = int(m * x + b)
        if y < 0 or y >= rows:
            break
        for yp in range(
                max(0, y - RANSAC_EPSILON), min(rows, y + RANSAC_EPSILON)):
            f += VdispImage[yp][x]
    return f


def getRANSACFittedLine(VdispImage):
    """Applies RANSAC to find the best line fit of the VDispImage. This is the
       line that fits the approximate road."""
    rows, cols = VdispImage.shape
    cumSumArray = np.cumsum(VdispImage)
    N = cumSumArray[-1]
    global bestM, bestB
    bestM *= BIN_SIZE  # Adjust m to VdispImage dimensions.
    bestF = evaluateRANSACTry(VdispImage, bestM, bestB)
    for i in range(RANSAC_TRIES):
        idx1 = np.searchsorted(cumSumArray, random.randint(1, N), side='left')
        idx2 = np.searchsorted(cumSumArray, random.randint(1, N), side='left')

        y1 = idx1 / cols
        x1 = idx1 - y1 * cols
        y2 = idx2 / cols
        x2 = idx2 - y2 * cols
        if x1 == x2:
            continue  # Do not consider vertical lines
        m = float(y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        f = evaluateRANSACTry(VdispImage, m, b)

        if f > bestF:
            bestF = f
            bestM = m
            bestB = b
    bestM /= BIN_SIZE  # Adjust m to original dispImage dimensions.
    return bestM, bestB
