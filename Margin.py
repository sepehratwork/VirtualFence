import numpy as np

def Margin(pts, k):
    '''
    Streches a polygon with corners 'pts' by k%
    '''
    k /= 100
    k += 1

    center = np.array([[0,0]])

    for i in range(len(pts)):
        center[0][0] = center[0][0] + pts[i][0][0]
        center[0][1] = center[0][1] + pts[i][0][1]

    center[0][0] = center[0][0] / len(pts)
    center[0][1] = center[0][1] / len(pts)

    margin = np.zeros(pts.shape)
    for i in range(len(margin)):
        vector = pts[i] - center
        margin[i] = (k * vector) + center
    
    return margin