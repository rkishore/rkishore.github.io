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
valid_three_imgs = (path/'valid'/'3').ls().sorted()
valid_seven_imgs = (path/'valid'/'7').ls().sorted()

valid_three_tensors_list = [tensor(Image.open(img)) for img in valid_three_imgs]
valid_seven_tensors_list = [tensor(Image.open(img)) for img in valid_seven_imgs]

valid_stacked_threes = torch.stack(valid_three_tensors_list).float()/255.
valid_stacked_sevens = torch.stack(valid_seven_tensors_list).float()/255.
valid_stacked_threes.shape, valid_stacked_sevens.shape
```




    (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))



```python
valid_x = torch.cat([valid_stacked_threes, valid_stacked_sevens]).view(-1, 28*28); valid_x.shape
```




    torch.Size([2038, 784])



```python
valid_y = tensor( [1]*len(valid_three_imgs) + [0]*len(valid_seven_imgs) ).unsqueeze(1); valid_y.shape
```




    torch.Size([2038, 1])



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

    0.7304 0.8506 0.9009 0.9292 0.9389 0.9443 0.9526 0.9546 0.9589 0.9614 

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
    0.5068 0.5068 0.5068 0.5068 0.5068 0.5068 0.5068 0.5068 0.5068 0.5068 

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




    tensor([-10.4003,   5.0197], grad_fn=<SelectBackward>)



```python
loss = cross_entropy_loss(preds, y); loss
```




    tensor(0.2297, grad_fn=<NllLossBackward>)



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




    tensor(0.0547)



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
    Mean loss:  tensor(0.3924)
    Accuracy:  0.6654
    Mean loss:  tensor(0.2471)
    Accuracy:  0.7858
    Mean loss:  tensor(0.1639)
    Accuracy:  0.8501
    Mean loss:  tensor(0.1191)
    Accuracy:  0.8878
    Mean loss:  tensor(0.0962)
    Accuracy:  0.9088
    Mean loss:  tensor(0.0831)
    Accuracy:  0.924
    Mean loss:  tensor(0.0746)
    Accuracy:  0.9303
    Mean loss:  tensor(0.0685)
    Accuracy:  0.9357
    Mean loss:  tensor(0.0638)
    Accuracy:  0.9416
    Mean loss:  tensor(0.0600)
    Accuracy:  0.9435


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
    0.6045 0.8349 0.9023 0.9228 0.9389 0.9501 0.956 0.9589 0.9633 0.9643 

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




    0.4748



```python
train_epoch(linear2, calc_grad_softmax_loss)
```




    tensor(0.5197)



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

    0.4932 0.8359 0.8418 0.9131 0.9331 0.9468 0.956 0.9629 0.9658 0.9668 

```python
# softmax_loss
lr = 0.1
linear3 = nn.Linear(28*28,2)
#opt = SGD(linear3.parameters(), lr)
opt = BasicOptim(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5215 0.8394 0.9248 0.9468 0.9604 0.9638 0.9648 0.9678 0.9687 0.9697 

A simple replacement we can make for our BasicOptim optimizer class is to replace it with the built-in SGD optimizer function

```python
# mnist_loss
lr = 1.
linear2 = nn.Linear(28*28,1)
opt = SGD(linear2.parameters(), lr)
train_model(linear2, 10, True)
```

    0.4932 0.7856 0.8555 0.9165 0.936 0.9502 0.958 0.9638 0.9658 0.9692 

```python
# softmax_loss
lr = 0.1
linear3 = nn.Linear(28*28,2)
opt = SGD(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5225 0.8496 0.9253 0.9443 0.9614 0.9653 0.9658 0.9678 0.9692 0.9702 

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
      <td>0.636797</td>
      <td>0.503552</td>
      <td>0.495584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.544573</td>
      <td>0.172633</td>
      <td>0.862610</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.199040</td>
      <td>0.192590</td>
      <td>0.823847</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.086776</td>
      <td>0.109772</td>
      <td>0.909225</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.045428</td>
      <td>0.079276</td>
      <td>0.931796</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.029346</td>
      <td>0.063142</td>
      <td>0.945535</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.022771</td>
      <td>0.053189</td>
      <td>0.955348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.019872</td>
      <td>0.046614</td>
      <td>0.962218</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.018415</td>
      <td>0.042011</td>
      <td>0.965653</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.017544</td>
      <td>0.038621</td>
      <td>0.966634</td>
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
      <td>0.149008</td>
      <td>0.390904</td>
      <td>0.531894</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.084927</td>
      <td>0.182335</td>
      <td>0.858685</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.058875</td>
      <td>0.104818</td>
      <td>0.928361</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.047090</td>
      <td>0.075925</td>
      <td>0.947988</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.040928</td>
      <td>0.061695</td>
      <td>0.960746</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.037252</td>
      <td>0.053434</td>
      <td>0.964671</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.034773</td>
      <td>0.048098</td>
      <td>0.965653</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.032933</td>
      <td>0.044379</td>
      <td>0.968597</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.031475</td>
      <td>0.041637</td>
      <td>0.968106</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.030271</td>
      <td>0.039525</td>
      <td>0.970069</td>
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
      <td>0.341516</td>
      <td>0.407349</td>
      <td>0.505888</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155696</td>
      <td>0.241218</td>
      <td>0.788027</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.084747</td>
      <td>0.118387</td>
      <td>0.912169</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.054734</td>
      <td>0.078711</td>
      <td>0.941119</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.040963</td>
      <td>0.060977</td>
      <td>0.955348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.034023</td>
      <td>0.051141</td>
      <td>0.964671</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.030093</td>
      <td>0.045006</td>
      <td>0.965653</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.027574</td>
      <td>0.040844</td>
      <td>0.966634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.025778</td>
      <td>0.037838</td>
      <td>0.969087</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024397</td>
      <td>0.035555</td>
      <td>0.969578</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.023281</td>
      <td>0.033749</td>
      <td>0.972522</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.022352</td>
      <td>0.032276</td>
      <td>0.974975</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.021561</td>
      <td>0.031041</td>
      <td>0.976448</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.020878</td>
      <td>0.029985</td>
      <td>0.976938</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.020279</td>
      <td>0.029068</td>
      <td>0.977920</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.019749</td>
      <td>0.028262</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.019274</td>
      <td>0.027546</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.018847</td>
      <td>0.026905</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.018458</td>
      <td>0.026327</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.018102</td>
      <td>0.025802</td>
      <td>0.978901</td>
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
      <td>0.263570</td>
      <td>0.433600</td>
      <td>0.504416</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.121355</td>
      <td>0.228203</td>
      <td>0.795388</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.066274</td>
      <td>0.115423</td>
      <td>0.908243</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.043652</td>
      <td>0.078083</td>
      <td>0.938175</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.033336</td>
      <td>0.060691</td>
      <td>0.953386</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.028139</td>
      <td>0.050785</td>
      <td>0.962218</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.025181</td>
      <td>0.044498</td>
      <td>0.966143</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.023261</td>
      <td>0.040209</td>
      <td>0.967615</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.021868</td>
      <td>0.037109</td>
      <td>0.968597</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.020781</td>
      <td>0.034763</td>
      <td>0.969578</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.019895</td>
      <td>0.032910</td>
      <td>0.971541</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.019153</td>
      <td>0.031399</td>
      <td>0.973013</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.018520</td>
      <td>0.030128</td>
      <td>0.973994</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.017971</td>
      <td>0.029036</td>
      <td>0.974485</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.017489</td>
      <td>0.028086</td>
      <td>0.975466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.017062</td>
      <td>0.027249</td>
      <td>0.976448</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.016677</td>
      <td>0.026508</td>
      <td>0.977429</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.016329</td>
      <td>0.025847</td>
      <td>0.977920</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.016011</td>
      <td>0.025255</td>
      <td>0.977429</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.015718</td>
      <td>0.024724</td>
      <td>0.977429</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


### Pretty cool accuracy numbers there! :-)
