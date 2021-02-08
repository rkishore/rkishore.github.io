# Exploring loss functions for the MNIST_SAMPLE dataset



**Objective:** In this notebook, I want to move towards classifying the full MNIST dataset of ten digits by first working with different loss functions for the smaller MNIST_SAMPLE dataset. The MNIST_SAMPLE dataset has data only for two digits (3s and 7s). The loss functions I explore are the `mnist_loss`, `softmax_loss`, and `cross_entropy_loss`. The reference for everything in this blog post is the [fastai 2020 course](https://course.fast.ai), especially the [amazing textbook](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).

In Chapter 4 of the textbook, we are taught how to get the data ready and to use the `mnist_loss` function that basically uses the `sigmoid` function on one column of activations from the final layer. Then, in Chapter 5, we are given examples of how to use the `softmax` function to achieve what the `sigmoid` function does, but on more than one column of activations. Moreover, the `cross_entropy_loss` function is introduced. This function basically adds a `log` to the `softmax` function, i.e. it does a `log_softmax` on the final layer of activations followed by selecting the loss corresponding to the column that corresponds to the target (using `nll_loss`). 

In this notebook, I want to learn how to use a loss function that will work with more than one column of activations from the final layer. With the MNIST_FULL dataset, my thinking is that it would be better to have ten columns of activations from the final layer as opposed to one really long column of activations. And beyond this, as per Chatper 5 in the textbook, we should use the `cross_entropy_loss` to aid classification among ten categories of images. Here, I want to understand the details of this planned approach and validate it with two digits before moving to classify data from ten digits.

### Get the basic imports out of the way

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
show_image(three_tensors_list[0])
```




    <AxesSubplot:>




![png](/images/04_mnist_basics_with_softmax_files/output_7_1.png)


```python
show_image(seven_tensors_list[0])
```




    <AxesSubplot:>




![png](/images/04_mnist_basics_with_softmax_files/output_8_1.png)


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
def linear1(xb, weights, bias): return xb@weights + bias # linear classifier with one weight per pixel
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

#### We see that the `batch_accuracy` function is not increasing at all. We need to debug what is going on here.

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


**Observation:** when we use softmax as a loss function, the behavior is as expected, i.e. the loss goes down and the `batch_accuracy` goes up. But with `cross_entropy_loss`, the `batch_accuracy` stays constant. Why? Needs to be investigated further. For now, we plan to use `softmax_loss` till we figure out why going forward with MNIST_FULL.

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

I could not get the traditional `cross_entropy_loss` using the negative log-likelihood function to work as expected with the MNIST_SAMPLE dataset of '3's and '7's and two columns of activations from the final layer. This is, in all likelihood, due to my inability to define the `batch_accuracy` function correctly? Using the `softmax_loss` alone on the other hand works as expected.

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

Lets redefine our loss functions to match the template expected by fastai and PyTorch. Lets also redefine the `train_epoch` function to use the optimizer defined above.

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
    return round(torch.stack(accs).mean().item(), 4)
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

    0.4888 0.8616 0.8242 0.9001 0.9282 0.944 0.9525 0.9581 0.9636 0.9676 

```python
# softmax_loss
lr = 0.1
linear3 = nn.Linear(28*28,2)
#opt = SGD(linear3.parameters(), lr)
opt = BasicOptim(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5217 0.8304 0.9157 0.9427 0.9552 0.9628 0.9674 0.9707 0.9723 0.9737 

A simple replacement we can make for our BasicOptim optimizer class is to replace it with the built-in SGD optimizer function

```python
# mnist_loss
lr = 1.
linear2 = nn.Linear(28*28,1)
opt = SGD(linear2.parameters(), lr)
train_model(linear2, 10, True)
```

    0.4888 0.7525 0.8579 0.9094 0.9333 0.9467 0.9546 0.9595 0.9648 0.9688 

```python
# softmax_loss
lr = 0.1
linear3 = nn.Linear(28*28,2)
opt = SGD(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5363 0.8386 0.9193 0.9441 0.955 0.9636 0.9671 0.9698 0.9718 0.9733 

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
    nn.Linear(30,1) # 1 column of activations from the final layer
)
```

```python
simple_net_softmax = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,2) # 2 columns of activations from the final layer
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
      <td>0.298082</td>
      <td>0.406346</td>
      <td>0.506050</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.140334</td>
      <td>0.229533</td>
      <td>0.802194</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.078882</td>
      <td>0.120482</td>
      <td>0.906986</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.052542</td>
      <td>0.082606</td>
      <td>0.937964</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.040197</td>
      <td>0.064589</td>
      <td>0.950307</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.033815</td>
      <td>0.054133</td>
      <td>0.959180</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.030114</td>
      <td>0.047274</td>
      <td>0.964343</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.027690</td>
      <td>0.042405</td>
      <td>0.967328</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.025933</td>
      <td>0.038752</td>
      <td>0.969990</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024566</td>
      <td>0.035897</td>
      <td>0.972088</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.023452</td>
      <td>0.033599</td>
      <td>0.973540</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.022520</td>
      <td>0.031709</td>
      <td>0.974992</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.021722</td>
      <td>0.030128</td>
      <td>0.976283</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.021032</td>
      <td>0.028784</td>
      <td>0.977251</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.020426</td>
      <td>0.027628</td>
      <td>0.977977</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.019889</td>
      <td>0.026624</td>
      <td>0.978541</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.019410</td>
      <td>0.025743</td>
      <td>0.978945</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.018977</td>
      <td>0.024963</td>
      <td>0.979348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.018585</td>
      <td>0.024268</td>
      <td>0.979671</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.018227</td>
      <td>0.023644</td>
      <td>0.980155</td>
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
      <td>0.248439</td>
      <td>0.425511</td>
      <td>0.505566</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.115231</td>
      <td>0.213843</td>
      <td>0.813892</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.063572</td>
      <td>0.113364</td>
      <td>0.907632</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.042388</td>
      <td>0.079075</td>
      <td>0.936189</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.032682</td>
      <td>0.062401</td>
      <td>0.950065</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.027737</td>
      <td>0.052443</td>
      <td>0.957567</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.024883</td>
      <td>0.045787</td>
      <td>0.963053</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.023011</td>
      <td>0.040991</td>
      <td>0.966925</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.021647</td>
      <td>0.037365</td>
      <td>0.969426</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.020583</td>
      <td>0.034523</td>
      <td>0.971846</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.019718</td>
      <td>0.032235</td>
      <td>0.973701</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.018994</td>
      <td>0.030358</td>
      <td>0.975073</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.018376</td>
      <td>0.028790</td>
      <td>0.976444</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.017839</td>
      <td>0.027465</td>
      <td>0.977251</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.017366</td>
      <td>0.026328</td>
      <td>0.977735</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.016946</td>
      <td>0.025345</td>
      <td>0.978219</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.016569</td>
      <td>0.024486</td>
      <td>0.978783</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.016228</td>
      <td>0.023730</td>
      <td>0.979590</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.015918</td>
      <td>0.023057</td>
      <td>0.980236</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.015633</td>
      <td>0.022458</td>
      <td>0.980800</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


### Pretty cool accuracy numbers there! :-)
