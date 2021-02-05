# Cross-entropy loss and the MNIST_SAMPLE dataset



**Objective:** In this notebook, I want to move towards classifying the full MNIST dataset by first using cross entropy loss or softmax as a loss function for the MNIST_SAMPLE dataset that has data only for two digits (3s and 7s), as opposed to the full ten digits in the MNIST dataset.

In Chapter 4 of the textbook, we are taught how to get the data ready and to use the mnist_loss function that basically uses the `sigmoid` function on one column of activations from the final layer. Then, in Chapter 5, we are given examples of how to use the `softmax` function to achieve what the `sigmoid` function does, but on more than one column of activations. Moreover, the cross entropy loss is introduced that basically does a `log_softmax` on the final layer of activations followed by selecting the loss corresponding to the column that corresponds to the target of interest (using `nll_loss`).

With the MNIST_FULL dataset, my thinking is that it would be better to have ten columns of activations from the final layer as opposed to one really long column of activations. And beyond this, we can use the cross entropy loss to aid classification among ten categories. Here, I want to understand the details of this planned approach and validate it with two digits before moving to classify data from ten digits.

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



### Prepare the dataset and dataloaders

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
# We define and use the softmax function as our loss as using the cross entropy function results in weird behavior
# We need to get to the bottom of this weird behavior
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

    0.7205 0.8401 0.8998 0.9242 0.9381 0.9464 0.9534 0.9573 0.9608 0.9628 

### Now let's use cross_entropy_loss and two columns of activations

```python
weights = init_params((28*28,2))
bias = init_params(2)
lr = 1.0
params = [weights, bias]
print(weights.shape, bias.shape)
for i in range(10):
    train_epoch(linear1, lr, params, calc_grad_ce_loss)
    print(validate_epoch(linear1, params, False), end=' ')
```

    torch.Size([784, 2]) torch.Size([2])
    0.5112 0.5112 0.5112 0.5112 0.5112 0.5112 0.5112 0.5112 0.5112 0.5112 

### Debug why the batch accuracy is not changing when we use cross entropy loss

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




    tensor([-1.7301,  4.6855], grad_fn=<SelectBackward>)



```python
loss = cross_entropy_loss(preds, y); loss
```




    tensor(1.0518, grad_fn=<NllLossBackward>)



```python
loss.backward()
```

```python
params = w,b
for p in params:
    #print(p.grad, p.grad.mean())
    p.data -= p.grad*lr
    p.grad.zero_()
```

```python
batch_accuracy(preds, y, False)
```




    tensor(0.2188)



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
    Mean loss:  tensor(0.6547)
    Accuracy:  0.425
    Mean loss:  tensor(0.5197)
    Accuracy:  0.5398
    Mean loss:  tensor(0.4005)
    Accuracy:  0.6602
    Mean loss:  tensor(0.2695)
    Accuracy:  0.7681
    Mean loss:  tensor(0.1824)
    Accuracy:  0.8331
    Mean loss:  tensor(0.1432)
    Accuracy:  0.8711
    Mean loss:  tensor(0.1168)
    Accuracy:  0.8934
    Mean loss:  tensor(0.1001)
    Accuracy:  0.9078
    Mean loss:  tensor(0.0886)
    Accuracy:  0.9191
    Mean loss:  tensor(0.0804)
    Accuracy:  0.9267


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
    0.6858 0.8494 0.913 0.9315 0.9449 0.9523 0.9563 0.9603 0.9619 0.9642 

### Conclusion

I could not get the traditional cross-entropy loss using the negative log-likelihood function to work well with the MNIST_SAMPLE dataset of '3's and '7's. Using softmax alone, on the other hand works well.

### Now, let's replace our code with PyTorch/fastai built-in functions and see if we get the same result

```python
linear2 = nn.Linear(28*28, 2)
w,b = linear2.parameters(); w.shape, b.shape
```




    (torch.Size([2, 784]), torch.Size([2]))



Lets put our step_weights function into a class

```python
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr
        
    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr
    
    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```

```python
opt = BasicOptim(linear2.parameters(), lr)
```

Lets redefine our train_epoch function to use the optimizer object above

```python
def softmax_loss(preds, tgts):
    preds = torch.softmax(preds, dim=1)
    #print(preds[0])
    idx = range(preds[:,0].shape[0])
    return preds[idx, torch.squeeze(tgts)].mean()
```

```python
def calc_grad_mnist_loss(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
    return loss
```

```python
def calc_grad_ce_loss(xb, yb, model):
    preds = model(xb)
    loss = cross_entropy_loss(preds, yb)
    #print(loss)
    loss.backward()
    return loss
```

```python
def calc_grad_softmax_loss(xb, yb, model):
    preds = model(xb)
    loss = softmax_loss(preds, yb)
    #print(loss)
    loss.backward()
    return loss
```

```python
def train_epoch(model, calc_grad_func):
    epoch_loss = []
    for xb,yb in dl:
        epoch_loss.append(calc_grad_func(xb,yb,model))
        opt.step()
        opt.zero_grad()
    epoch_loss_tnsr = tensor(epoch_loss)
    return epoch_loss_tnsr.mean()
```

We need to redefine the validate_epoch function simply because we don't need to pass the params as args to the built-in Pytorch function

```python
def validate_epoch_new(model, mnist=True):
    accs = [batch_accuracy(model(xb), yb, mnist) for xb,yb in valid_dl]
    #if mnist:
    return round(torch.stack(accs).mean().item(), 4)
    #else:
    #    accs_tensor = tensor(accs)
        #print(len(accs), accs[0], accs_tensor[0], accs_tensor.mean())
    #    accuracy = round(accs_tensor.mean().item(), 4)
        #print("Accuracy: ", accuracy)
    #    return accuracy
```

```python
validate_epoch_new(linear2, False)
```




    0.6437



```python
train_epoch(linear2, calc_grad_softmax_loss)
```

```python
def train_model(model, epochs, mnist=True):
    for i in range(epochs):
        if mnist:
            epoch_loss_mean = train_epoch(model, calc_grad_mnist_loss)
        else:
            epoch_loss_mean = train_epoch(model, calc_grad_softmax_loss)
        #print(epoch_loss_mean)
        print(validate_epoch_new(model, mnist), end=' ')
```

```python
# mnist_loss
lr = 1.
linear2 = nn.Linear(28*28,1)
#opt = SGD(linear2.parameters(), lr)
opt = BasicOptim(linear2.parameters(), lr)
train_model(linear2, 10, True)
```

    0.4888 0.7474 0.8562 0.9078 0.9326 0.9458 0.9539 0.9594 0.9644 0.9685 

```python
# softmax_loss
lr = 0.1
linear3 = nn.Linear(28*28,2)
#opt = SGD(linear3.parameters(), lr)
opt = BasicOptim(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5418 0.8451 0.9199 0.944 0.9561 0.963 0.9669 0.9696 0.9723 0.9741 

A simple replacement we can make for our BasicOptim optimizer class is to replace it with the built-in SGD optimizer function

```python
# mnist_loss
lr = 1.
linear2 = nn.Linear(28*28,1)
opt = SGD(linear2.parameters(), lr)
train_model(linear2, 10, True)
```

    0.4888 0.7524 0.8551 0.9088 0.933 0.9463 0.9541 0.959 0.9645 0.9684 

```python
# softmax_loss
lr = 0.1
linear3 = nn.Linear(28*28,2)
opt = SGD(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5283 0.8414 0.92 0.9442 0.9561 0.9633 0.9673 0.9705 0.9724 0.9738 

Instead of `train_model`, we can now transition over to using the built-in `Learner.fit` class method. Before we do this, lets redefine batch_accuracy for mnist and softmax separately as there is a need to follow the template here for this method

```python
def batch_accuracy_mnist(xb,yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```

```python
def batch_accuracy_softmax(xb,yb):
    preds = torch.softmax(xb, dim=1)
    yb_squeezed = torch.squeeze(yb)
    #print(xb.shape, yb.shape, preds.shape, yb_squeezed.shape)
    correct = (preds[:,0]>0.5) == yb_squeezed
    return correct.float().mean()
```

```python
learn_mnist = Learner(dls, nn.Linear(28*28,1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy_mnist)
```

```python
learn_softmax = Learner(dls, nn.Linear(28*28,2), opt_func=SGD, loss_func=softmax_loss, metrics=batch_accuracy_softmax)
```

```python
learn_mnist.fit(10, lr=1.0)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy_mnist</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.637177</td>
      <td>0.504163</td>
      <td>0.494595</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.506652</td>
      <td>0.218602</td>
      <td>0.804453</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.187165</td>
      <td>0.177797</td>
      <td>0.836641</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.082407</td>
      <td>0.109837</td>
      <td>0.902065</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.043772</td>
      <td>0.081640</td>
      <td>0.928929</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.028667</td>
      <td>0.065939</td>
      <td>0.944256</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.022442</td>
      <td>0.055925</td>
      <td>0.952323</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.019662</td>
      <td>0.048953</td>
      <td>0.958454</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.018244</td>
      <td>0.043798</td>
      <td>0.963779</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.017387</td>
      <td>0.039850</td>
      <td>0.967570</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
learn_softmax.fit(10, lr=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy_softmax</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.164560</td>
      <td>0.393483</td>
      <td>0.524685</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.089713</td>
      <td>0.191804</td>
      <td>0.841884</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.060771</td>
      <td>0.111265</td>
      <td>0.918361</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.047979</td>
      <td>0.080447</td>
      <td>0.942965</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.041427</td>
      <td>0.064905</td>
      <td>0.954743</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.037578</td>
      <td>0.055631</td>
      <td>0.962085</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.035008</td>
      <td>0.049473</td>
      <td>0.966602</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.033112</td>
      <td>0.045073</td>
      <td>0.970152</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.031616</td>
      <td>0.041761</td>
      <td>0.971442</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.030383</td>
      <td>0.039169</td>
      <td>0.973459</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


### Using a neural network instead of a linear classifier

```python
simple_net_mnist = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```

```python
simple_net_softmax = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,2)
)
```

```python
learn_mnist = Learner(dls, simple_net_mnist, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy_mnist)
```

```python
learn_softmax = Learner(dls, simple_net_softmax, opt_func=SGD, loss_func=softmax_loss, metrics=batch_accuracy_softmax)
```

```python
learn_mnist.fit(20, 0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy_mnist</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.340772</td>
      <td>0.400088</td>
      <td>0.507180</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.157275</td>
      <td>0.240046</td>
      <td>0.790094</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.085858</td>
      <td>0.122534</td>
      <td>0.906986</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.055504</td>
      <td>0.082693</td>
      <td>0.937883</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.041521</td>
      <td>0.064401</td>
      <td>0.950871</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.034447</td>
      <td>0.053921</td>
      <td>0.959019</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.030429</td>
      <td>0.047091</td>
      <td>0.964585</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.027849</td>
      <td>0.042260</td>
      <td>0.967247</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.026009</td>
      <td>0.038632</td>
      <td>0.969587</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024594</td>
      <td>0.035798</td>
      <td>0.971926</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.023454</td>
      <td>0.033514</td>
      <td>0.973136</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.022505</td>
      <td>0.031634</td>
      <td>0.974669</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.021700</td>
      <td>0.030058</td>
      <td>0.976121</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.021005</td>
      <td>0.028717</td>
      <td>0.976847</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.020397</td>
      <td>0.027564</td>
      <td>0.978057</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.019860</td>
      <td>0.026560</td>
      <td>0.978945</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.019380</td>
      <td>0.025680</td>
      <td>0.979348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.018947</td>
      <td>0.024902</td>
      <td>0.979348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.018554</td>
      <td>0.024208</td>
      <td>0.980074</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.018195</td>
      <td>0.023585</td>
      <td>0.980639</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
learn_softmax.fit(20, 0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy_softmax</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.252010</td>
      <td>0.427179</td>
      <td>0.505486</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.116402</td>
      <td>0.219587</td>
      <td>0.805744</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.063833</td>
      <td>0.114854</td>
      <td>0.907228</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.042337</td>
      <td>0.079503</td>
      <td>0.936673</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.032580</td>
      <td>0.062424</td>
      <td>0.949903</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.027662</td>
      <td>0.052312</td>
      <td>0.957325</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.024847</td>
      <td>0.045597</td>
      <td>0.963537</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.023005</td>
      <td>0.040788</td>
      <td>0.966441</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.021659</td>
      <td>0.037145</td>
      <td>0.970313</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.020604</td>
      <td>0.034299</td>
      <td>0.972572</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.019738</td>
      <td>0.032012</td>
      <td>0.974024</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.019009</td>
      <td>0.030134</td>
      <td>0.975395</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.018383</td>
      <td>0.028569</td>
      <td>0.976847</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.017836</td>
      <td>0.027249</td>
      <td>0.977735</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.017352</td>
      <td>0.026121</td>
      <td>0.978380</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.016921</td>
      <td>0.025147</td>
      <td>0.978783</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.016533</td>
      <td>0.024297</td>
      <td>0.979106</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.016181</td>
      <td>0.023550</td>
      <td>0.979590</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.015860</td>
      <td>0.022887</td>
      <td>0.980155</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.015565</td>
      <td>0.022295</td>
      <td>0.980800</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


#### Pretty cool accuracy numbers there! :-)
