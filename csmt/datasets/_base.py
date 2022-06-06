
import numpy as np
import gzip

def load_digits_zhs():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data
    y = digits.target
    mask=get_true_mask([column for column in X])
    return X,y,mask

def gen_tore_vecs(dims, number, rmin, rmax):
    vecs = np.random.uniform(low=-1, size=(number,dims))
    radius = rmin + np.random.sample(number) * (rmax-rmin)
    mags = np.sqrt((vecs*vecs).sum(axis=-1))
    # How to distribute the magnitude to the vectors
    for i in range(number):
        vecs[i,:] = vecs[i, :] / mags[i] *radius[i]
    return vecs[:,0], vecs[:,1]

def load_donut():
    print('build donnuts data')
    Nobjs = 2000
    xn, yn = gen_tore_vecs(2, Nobjs, 1.5, 4)
    Xn = np.array([xn, yn]).T

    Nobjsb = 1000
    mean = [0, 0]
    cov = [[.5, 0], [0, .5]]  # diagonal covariance
    xb, yb = np.random.multivariate_normal(mean, cov, Nobjsb).T
    Xb = np.array([xb, yb]).T
    
    # create cluster of anomalies
    mean = [3., 3.]
    cov = [[.25, 0], [0, .25]]  # diagonal covariance
    Nobjsa = 1000
    xa, ya = np.random.multivariate_normal(mean, cov, Nobjsa).T
    Xa = np.array([xa, ya]).T
    
    Xab=np.concatenate([Xa,Xb])
    yn=np.zeros(Xn.shape[0])
    yab=np.ones(Xab.shape[0])

    X=np.concatenate([Xn,Xab])
    y=np.concatenate([yn,yab])
    y=y.astype(np.int32)
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_blob():
    from csmt.datasets.data_generator.Symbolic_regression_classification_generator import gen_regression_symbolic, gen_classification_symbolic
    random_state = 999
    n_features = 2 # Number of features
    n_samples = 1500  # Number of samples
    centers = [[-2, 0], [2, -2], [2, 2]]  # Centers of the clusters
    cluster_std = 0.8  # Standard deviation of the clusters
    center_box=(-10.0, 10.0)

    # X, y = make_blobs(
    # n_samples=n_samples,
    # n_features=n_features,
    # centers=centers,
    # cluster_std=cluster_std,
    # center_box=center_box,
    # random_state=random_state)
    # X, y = make_classification(n_features=n_features,n_samples=1000,random_state=42)
    # X, y=make_classification(n_samples=n_samples,n_features=n_features,n_informative=2,n_redundant=0,n_clusters_per_class=2,random_state=2)
    # fun_t='((x1^2)/3-(x2^2)/15)'
    # fun_t='x1-40*x2+20'
    fun_t='x1-x2'
    data = gen_classification_symbolic(m=fun_t,n_samples= n_samples,n_features=n_features,flip_y=0)
    X=data[:,0:-1]
    y=data[:,-1]
    # X,y=make_circles(n_samples=n_samples,shuffle=True,noise=0.2,random_state=0,factor=0.5)
    # X,y=make_moons(n_samples=n_samples,shuffle=True,noise=0.2,random_state=0)
    # X,y=make_gaussian_quantiles(n_samples=n_samples,n_features=n_features,n_classes=2)

    mask=get_true_mask([column for column in X])
    return X,y,mask
    


def load_iris_zhs():
    num_classes = 2
    X, y = get_data(num_classes=num_classes)   
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_breast_cancer_zhs():
    from sklearn.datasets import load_breast_cancer
    cancer=load_breast_cancer()
    X=cancer.data
    y=cancer.target
    # X=X[:,0:10]
    mask=get_true_mask([column for column in X])
    return X,y,mask

# def load_mnist_flat_zhs():
#     (X_train, y_train), (X_test, y_test), min_, max_ = load_mnist()

#     n_samples_train = X_train.shape[0]
#     n_features_train = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
#     n_samples_test = X_test.shape[0]
#     n_features_test = X_test.shape[1] * X_test.shape[2] * X_test.shape[3]

#     X_train = X_train.reshape(n_samples_train, n_features_train)
#     X_test = X_test.reshape(n_samples_test, n_features_test)

#     y_train = np.argmax(y_train, axis=1)
#     y_test = np.argmax(y_test, axis=1)
#     X=X_train
#     y=y_train
#     mask=get_true_mask([column for column in X])
#     return X,y,mask


def load_mnist():
    # import torchvision
    # from torchvision import transforms
    # transformer = transforms.Compose([transforms.ToTensor()])
    # train_set = torchvision.datasets.MNIST(root='csmt/datasets/data/mnist/', train=True, transform=transformer, download=True)
    # test_set = torchvision.datasets.MNIST(root='csmt/datasets/data/mnist/', train=False, transform=transformer, download=True)
    train_data = extract_data("csmt/datasets/data/mnist/MNIST/raw/train-images-idx3-ubyte.gz", 60000)
    train_labels = extract_labels("csmt/datasets/data/mnist/MNIST/raw/train-labels-idx1-ubyte.gz", 60000)
    test_data = extract_data("csmt/datasets/data/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz", 10000)
    test_labels = extract_labels("csmt/datasets/data/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz", 10000)

    VALIDATION_SIZE = 5000
    
    validation_data = train_data[:VALIDATION_SIZE, :, :, :]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, :, :, :]
    train_labels = train_labels[VALIDATION_SIZE:]
    mask=get_true_mask([column for column in train_data])

    return train_data,train_labels, validation_data,validation_labels,test_data,test_labels,mask



def get_data(num_classes):
    from sklearn.datasets import load_iris
    x_train, y_train = load_iris(return_X_y=True)
    x_train = x_train[y_train < num_classes][:, [0, 1]]
    y_train = y_train[y_train < num_classes]
    x_train[:, 0][y_train == 0] *= 2
    x_train[:, 1][y_train == 2] *= 2
    x_train[:, 0][y_train == 0] -= 3
    x_train[:, 1][y_train == 2] -= 2
    
    x_train[:, 0] = (x_train[:, 0] - 4) / (9 - 4)
    x_train[:, 1] = (x_train[:, 1] - 1) / (6 - 1)
    
    return x_train, y_train 

def get_mask(orig_array,allow_array):
    mask_array=[]
    for item in orig_array:
        if item in allow_array:
            mask_array.append(True)
        else:
            mask_array.append(False)
    mask=np.array(mask_array)
    return mask

def get_true_mask(orig_array):
    mask_array=[]
    for item in orig_array:
        mask_array.append(True)
    mask=np.array(mask_array)
    return mask

def get_dict(dict_val):
    dic_={}
    i=0
    for key,value in dict_val:
        i=i+1
        dic_[key]=i
    return dic_

def add_str(x):
    x_arr=x.split(',')
    while len(x_arr) < 20:
        x_arr.append('0')
    x_out = ','.join(x_arr)
    return x_out

# def train_val_test_split(X,y, ratio_train, ratio_test, ratio_val):
#     X_train,X_middle,y_train,y_middle = train_test_split(X,y, stratify=y,train_size=ratio_train, test_size=ratio_test + ratio_val,random_state=42)
#     ratio = ratio_val/(1-ratio_train)
#     X_test,X_val,y_test,y_val = train_test_split(X_middle,y_middle,stratify=y_middle,test_size=ratio,random_state=42)
#     return X_train,y_train,X_val,y_val,X_test,y_test


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # data = (data / 255) - 0.5
        data = data.reshape(num_images, 1, 28, 28)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels