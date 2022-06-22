import numpy as np

                       # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
lut_ideal_15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0],   # 0
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1,  1,  1,  1],   # 1
                         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  2,  2,  2,  2],   # 2
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,  2,  2,  3,  3,  3],   # 3
                         [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,  3,  3,  3,  4,  4],   # 4
                         [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,  4,  4,  4,  5,  5],   # 5
                         [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4,  4,  5,  5,  6,  6],   # 6
                         [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,  5,  6,  6,  7,  7],   # 7
                         [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,  6,  6,  7,  7,  8],   # 8
                         [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6,  7,  7,  8,  8,  9],   # 9
                         [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7,  7,  8,  9,  9,  10],  # 10
                         [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7,  8,  9,  10, 10, 11],  # 11
                         [0, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8,  9,  10, 10, 11, 12],  # 12
                         [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9,  10, 10, 11, 12, 13],  # 13
                         [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9,  10, 11, 12, 13, 14],  # 14
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]) # 15

                        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15]])
lut_actual_15 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0],   # 0
                          [0, 1, 1, 1, 1, 1, 2, 2, 2, 2,  2,  3,  3,  3,  3,  3],   # 1
                          [0, 1, 2, 2, 2, 3, 3, 3, 4, 4,  4,  5,  5,  5,  5,  6],   # 2
                          [0, 2, 2, 3, 3, 3, 4, 4, 5, 5,  5,  6,  6,  7,  7,  7],   # 3
                          [0, 2, 3, 3, 4, 4, 5, 5, 6, 6,  6,  7,  7,  8,  8,  8],   # 4
                          [0, 2, 3, 3, 4, 5, 5, 6, 6, 7,  7,  8,  8,  9,  9,  9],   # 5
                          [0, 2, 3, 4, 4, 5, 6, 6, 7, 7,  8,  8,  9,  9,  10, 10],  # 6
                          [0, 3, 3, 4, 5, 5, 6, 7, 7, 8,  8,  9,  9,  10, 10, 11],  # 7
                          [0, 3, 3, 4, 5, 6, 6, 7, 8, 8,  9,  9,  10, 10, 11, 12],  # 8
                          [0, 3, 4, 4, 5, 6, 7, 7, 8, 9,  9,  10, 10, 11, 12, 12],  # 9
                          [0, 3, 4, 5, 5, 6, 7, 8, 8, 9,  10, 10, 11, 12, 12, 13],  # 10
                          [0, 3, 4, 5, 5, 6, 7, 8, 8, 9,  10, 11, 11, 12, 12, 13],  # 11
                          [0, 3, 4, 5, 6, 6, 7, 8, 9, 9,  10, 11, 12, 12, 13, 13],  # 12
                          [0, 3, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14],  # 13
                          [0, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 14],  # 14
                          [0, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15]]) # 15