Header header

# Slope: rows/disp; the row change per disparity increase.
float32 m
# Intersection: Corresponds to the horizon row.
float32 b
# For a disparity 'd', "d * m + b" is the row value of a pixel if it is in the
# road plane.
# For a given row 'r', "(r - b) / m" would be the expected disparity value for a
# pixel in that row to be part of the road plane.
