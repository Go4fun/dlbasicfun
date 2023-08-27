import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    #return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    print(np.sum(t*y))
    return -np.sum(t * np.log(y+1e-7))/ batch_size

if __name__ == '__main__':
    y = np.array([[0.1, 0.05, 0.6]
                  ,[0.0, 0.0, 1.0]
                  ])
    t = np.array([[0, 0, 1]
                  ,[1, 0, 0]
                  ])
    print(cross_entropy_error(y, t))

    x = np.array([[0.6, 0.9], [0.0, 0.1], [0.7, 0.2]])
    print(np.sum(x))