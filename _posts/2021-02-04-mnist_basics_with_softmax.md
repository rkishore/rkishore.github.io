# Title



```python
from fastai.vision.all import *
```

```python
import fastbook
fastbook.setup_book()
from fastbook import *
```

### Get your data ready

```python
path = untar_data(URLs.MNIST_SAMPLE)
path.ls()
```




    (#4) [Path('/home/igolgi/.fastai/data/mnist_sample/valid'),Path('/home/igolgi/.fastai/data/mnist_sample/labels.csv'),Path('/home/igolgi/.fastai/data/mnist_sample/train'),Path('/home/igolgi/.fastai/data/mnist_sample/models')]



```python
three_imgs = (path/'train'/'3').ls().sorted()
seven_imgs = (path/'train'/'7').ls().sorted()

three_tensors_list = [tensor(Image.open(img)) for img in three_imgs]
seven_tensors_list = [tensor(Image.open(img)) for img in seven_imgs]

stacked_threes = torch.stack(three_tensors_list).float()/255.
stacked_sevens = torch.stack(seven_tensors_list).float()/255.
stacked_threes.shape, stacked_sevens.shape
```




    (torch.Size([6131, 28, 28]), torch.Size([6265, 28, 28]))



```python
train_x = torch.cat([stacked_threes, stacked_sevens]); train_x.shape
```




    torch.Size([12396, 28, 28])



```python
train_x = train_x.view(-1, 28*28); train_x.shape
```




    torch.Size([12396, 784])



```python
train_y = tensor( [1]*len(three_imgs) + [0]*len(seven_imgs) ).unsqueeze(1); train_y.shape
```




    torch.Size([12396, 1])



```python
valid_three_imgs = (path/'train'/'3').ls().sorted()
valid_seven_imgs = (path/'train'/'7').ls().sorted()

valid_three_tensors_list = [tensor(Image.open(img)) for img in valid_three_imgs]
valid_seven_tensors_list = [tensor(Image.open(img)) for img in valid_seven_imgs]

valid_stacked_threes = torch.stack(valid_three_tensors_list).float()/255.
valid_stacked_sevens = torch.stack(valid_seven_tensors_list).float()/255.
valid_stacked_threes.shape, valid_stacked_sevens.shape
```




    (torch.Size([6131, 28, 28]), torch.Size([6265, 28, 28]))



```python
valid_x = torch.cat([valid_stacked_threes, valid_stacked_sevens]).view(-1, 28*28); valid_x.shape
```




    torch.Size([12396, 784])



```python
valid_y = tensor( [1]*len(valid_three_imgs) + [0]*len(valid_seven_imgs) ).unsqueeze(1); valid_y.shape
```




    torch.Size([12396, 1])



#### Prepare the dataset and dataloaders

```python
dset = list(zip(train_x, train_y))
valid_dset = list(zip(valid_x, valid_y))
```

```python
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)
dls = DataLoaders(dl, valid_dl)
```

### Get a basic linear classifier going as well as the loss functions, optimizers and metric functions

```python
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
```

```python
def linear1(xb, weights, bias): return xb@weights + bias
```

```python
def mnist_loss(preds, tgts):
    preds = preds.sigmoid()
    return torch.where(tgts==1, 1-preds, preds).mean()
```

```python
def cross_entropy_loss(preds, tgts):
    preds = torch.softmax(preds, dim=1)
    log_preds = torch.log(preds)
    #print(preds.shape, preds[0], preds[:,0].shape[0])
    return F.nll_loss(log_preds, torch.squeeze(tgts))
```

```python
def softmax_loss(preds, tgts):
    preds = torch.softmax(preds, dim=1)
    idx = range(preds[:,0].shape[0])
    return preds[idx, torch.squeeze(tgts)].mean()
```

```python
def calc_grad_mnist_loss(xb, yb, model, params):
    preds = model(xb, params[0], params[1])
    loss = mnist_loss(preds, yb)
    loss.backward()
```

```python
def calc_grad_ce_loss(xb, yb, model, params):
    preds = model(xb, params[0], params[1])
    loss = cross_entropy_loss(preds, yb)
    #print(loss)
    loss.backward()
```

```python
def calc_grad_softmax_loss(xb, yb, model, params):
    preds = model(xb, params[0], params[1])
    loss = softmax_loss(preds, yb)
    #print(loss)
    loss.backward()
```

```python
def step_weights(params, lr):
    for p in params:
        p.data -= p.grad*lr
        p.grad.zero_()
```

```python
def train_epoch(model, lr, params, calc_grad_func):
    for xb,yb in dl:
        calc_grad_func(xb,yb,model,params)
        step_weights(params,lr)
```

```python
def batch_accuracy(xb,yb, mnist=True):
    if mnist:
        preds = xb.sigmoid()
        correct = (preds>0.5) == yb
    else:
        preds = torch.softmax(xb, dim=1)
        yb_squeezed = torch.squeeze(yb)
        #print(xb.shape, yb.shape, preds.shape, yb_squeezed.shape)
        correct = (preds[:,0]>0.5) == yb_squeezed
    return correct.float().mean()
```

```python
def validate_epoch(model, params, mnist=True):
    accs = [batch_accuracy(model(xb, params[0], params[1]), yb, mnist) for xb,yb in valid_dl]
    if mnist:
        return round(torch.stack(accs).mean().item(), 4)
    else:
        accs_tensor = tensor(accs)
        #print(len(accs), accs[0], accs_tensor[0], accs_tensor.mean())
        accuracy = round(accs_tensor.mean().item(), 4)
        #print("Accuracy: ", accuracy)
        return accuracy
```

### Now, lets use mnist_loss and one column of activations

```python
weights = init_params((28*28,1))
bias = init_params(1)
lr = 1.0
params = [weights, bias]
for i in range(10):
    train_epoch(linear1, lr, params, calc_grad_mnist_loss)
    print(validate_epoch(linear1, params), end=' ')
```

    0.6541 0.8476 0.9062 0.9316 0.9447 0.9539 0.9588 0.9616 0.9638 0.966 

### Now let's use cross_entropy_loss and two columns of activations

```python
weights = init_params((28*28,2))
bias = init_params(2)
lr = 1.0
params = [weights, bias]
print(weights.shape, bias.shape)
for i in range(10):
    train_epoch(linear1, lr, params, calc_grad_softmax_loss)
    print(validate_epoch(linear1, params, False), end=' ')
```

    torch.Size([784, 2]) torch.Size([2])
    0.8408 0.9014 0.9273 0.9442 0.953 0.9581 0.963 0.9664 0.9684 0.9699 

### Debug the batch accuracy not changing

```python
x,y = dl.one_batch()
```

```python
x.shape, y.shape
```




    (torch.Size([256, 784]), torch.Size([256, 1]))



```python
w = init_params((28*28,2)); b = init_params(2); w.shape, b.shape
```




    (torch.Size([784, 2]), torch.Size([2]))



```python
preds = linear1(x, w, b); preds[:,0].shape[0]
```




    256



```python
preds[0]
```




    tensor([4.4004, 3.5669], grad_fn=<SelectBackward>)



```python
loss = cross_entropy_loss(preds, y); loss
```

    torch.Size([256, 2]) tensor([0.6971, 0.3029], grad_fn=<SelectBackward>)





    tensor(0.4793, grad_fn=<MeanBackward0>)



```python
loss.backward()
```

```python
params = w,b
for p in params:
    print(p.grad, p.grad.mean())
    p.data -= p.grad*lr
    p.grad.zero_()
```

    tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            ...,
            [0., 0.],
            [0., 0.],
            [0., 0.]]) tensor(7.6026e-11)
    tensor([-0.0449,  0.0449]) tensor(1.8626e-09)


```python
batch_accuracy(preds, y, False)
```




    tensor(0.5234)



```python
w = init_params((28*28,2)); b = init_params(2); print(w.shape, b.shape)
lr = 0.1
```

    torch.Size([784, 2]) torch.Size([2])


```python
def softmax_loss(preds, tgts):
    preds = torch.softmax(preds, dim=1)
    #print(preds.shape, preds[0], preds[:,0].shape[0])
    idx = range(preds[:,0].shape[0])
    return preds[idx, torch.squeeze(tgts)].mean()
```

```python
def cross_entropy_loss(preds, tgts):
    preds = torch.softmax(preds, dim=1)
    log_preds = torch.log(preds)
    #print(preds.shape, preds[0], preds[:,0].shape[0])
    #return F.nll_loss(log_preds, torch.squeeze(tgts)) # This is the culprit!
    idx = range(log_preds[:,0].shape[0])
    return log_preds[idx, torch.squeeze(tgts)].mean()
```

```python
def calc_grad_single(x1, y1, w1, b1):
    preds = linear1(x1, w1, b1);
    loss = softmax_loss(preds, y1); 
    #print("Loss: ", loss)
    loss.backward()
    return loss
```

```python
def step_weights_single(p):
    p_grad_mean = p.grad.mean()
    p.data -= p.grad*lr
    p.grad.zero_()
    return p_grad_mean
```

```python
def train_epoch_single():
    max_batches = 5
    cur_batch = 0
    batch_loss = []
    for xb, yb in dl:
        batch_loss.append(calc_grad_single(xb, yb, w, b))
        w_grad_mean = step_weights_single(w)
        b_grad_mean = step_weights_single(b)
        #print("Weights.grad.mean and bias.grad.mean: ", w_grad_mean, b_grad_mean)
        cur_batch += 1
        #if cur_batch >= max_batches:
        #    break
    batch_loss_tensor = tensor(batch_loss)
    print("Mean loss: ", batch_loss_tensor.mean())
```

```python
def batch_accuracy_single(xb, yb):
    preds = torch.softmax(xb, dim=1)
    yb_squeezed = torch.squeeze(yb)
    #print(xb.shape, yb.shape, preds.shape, yb_squeezed.shape)
    correct = (preds[:,0]>0.5) == yb_squeezed
    return correct.float().mean()
```

```python
def validate_epoch_single(model):
    accs = [batch_accuracy_single(model(xb,w,b), yb) for xb,yb in valid_dl]
    accs_tensor = tensor(accs)
    #print(len(accs), accs[0], accs_tensor[0], accs_tensor.mean())
    accuracy = round(accs_tensor.mean().item(), 4)
    print("Accuracy: ", accuracy)
    return accuracy
```

```python
w = init_params((28*28,2)); b = init_params(2); print(w.shape, b.shape)
lr = 0.1
for i in range(10):
    train_epoch_single()
    validate_epoch_single(linear1)
```

    torch.Size([784, 2]) torch.Size([2])
    Mean loss:  tensor(0.5159)
    Accuracy:  0.5184
    Mean loss:  tensor(0.4597)
    Accuracy:  0.564
    Mean loss:  tensor(0.4149)
    Accuracy:  0.6142
    Mean loss:  tensor(0.3543)
    Accuracy:  0.6817
    Mean loss:  tensor(0.2811)
    Accuracy:  0.7528
    Mean loss:  tensor(0.2122)
    Accuracy:  0.812
    Mean loss:  tensor(0.1665)
    Accuracy:  0.8474
    Mean loss:  tensor(0.1383)
    Accuracy:  0.8743
    Mean loss:  tensor(0.1195)
    Accuracy:  0.8896
    Mean loss:  tensor(0.1066)
    Accuracy:  0.9016


## Could not get the traditional cross-entropy loss using the negative log-likelihood function to work well with the MNIST_SAMPLE dataset of '3's and '7's. Using softmax alone, on the other hand works well.
