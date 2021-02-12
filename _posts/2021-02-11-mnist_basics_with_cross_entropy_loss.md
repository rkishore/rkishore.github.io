# Using cross entropy loss for the MNIST_SAMPLE dataset



**Objective:** As in the last notebook, I want to move towards classifying the full MNIST dataset of ten digits by first working with loss functions that can work with N activations from the final layer for the smaller MNIST_SAMPLE dataset. The MNIST_SAMPLE dataset has data only for two digits (3s and 7s). The loss functions I explored previously were the `mnist_loss`, `softmax_loss`, and `cross_entropy_loss`. However, at that time, I had been unable to successfully use `cross_entropy_loss` for this dataset and assumed that we could use `softmax_loss` instead as it seemed to work well.

Turns out that I was defining the `batch_accuracy` function incorrectly then. I learnt this the hard way when I tried "scaling up" from two to ten categories. The `softmax_loss` function was just not able to push the accuracy up. After some head-scratching and googling, I realized the book did not teach us precisely how to calculate the `batch_accuracy` for the case where the last layer has N activations corresponding to N categories. Fortunately, it is not that hard and I found a good reference [here](https://jonathan-sands.com/deep%20learning/fastai/pytorch/vision/classifier/2020/11/15/MNIST.html#Training-a-neural-network)

As before, the reference for everything in this blog post is the [fastai 2020 course](https://course.fast.ai), especially the [amazing textbook](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).

In this notebook, I learn how to use the `cross_entropy_loss` for the MNIST_SAMPLE dataset with two categories. Scaling up to ten categories from this point is straight-forward and the topic of my next notebook. 

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




![png](/images/04_mnist_basics_with_cross_entropy_files/output_7_1.png)


```python
show_image(seven_tensors_list[0])
```




    <AxesSubplot:>




![png](/images/04_mnist_basics_with_cross_entropy_files/output_8_1.png)


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
    preds = torch.log_softmax(preds, dim=1)
    #print(preds.shape, preds[0], preds[:,0].shape[0])
    return F.nll_loss(preds, torch.squeeze(tgts))
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
        _,preds = torch.max(xb, 1)
        yb_squeezed = torch.squeeze(yb)
        #print(xb.shape, yb.shape, preds.shape, yb_squeezed.shape)
        correct = (preds == yb_squeezed)
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
lr = 1e-3 # Note the low learning rate here
params = [weights, bias]
print(weights.shape, bias.shape)
for i in range(10):
    train_epoch(linear1, lr, params, calc_grad_ce_loss)
    print(validate_epoch(linear1, params, False), end=' ')
```

    torch.Size([784, 2]) torch.Size([2])
    0.7228 0.7372 0.7526 0.7689 0.7812 0.7877 0.7995 0.8064 0.8114 0.8163 

### Looking at a single batch

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




    tensor(0.9453)



### Conclusion

I am able to use the traditional `cross_entropy_loss` using the negative log-likelihood function to work as expected with the MNIST_SAMPLE dataset of '3's and '7's and two columns of activations from the final layer. In the previous notebook, I could not do so due to my inability to define the `batch_accuracy` function correctly.

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




    0.6663



```python
train_epoch(linear2, calc_grad_ce_loss)
```




    tensor(0.5787)



```python
def train_model(model, epochs, mnist=True):
    for i in range(epochs):
        if mnist:
            epoch_loss_mean = train_epoch(model, calc_grad_mnist_loss)
        else:
            epoch_loss_mean = train_epoch(model, calc_grad_ce_loss)
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

    0.4932 0.8047 0.8452 0.9126 0.9336 0.9468 0.9551 0.9619 0.9663 0.9668 

```python
# softmax_loss
lr = 1e-2
linear3 = nn.Linear(28*28,2)
#opt = SGD(linear3.parameters(), lr)
opt = BasicOptim(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5269 0.8877 0.938 0.955 0.9633 0.9643 0.9638 0.9663 0.9658 0.9668 

A simple replacement we can make for our BasicOptim optimizer class is to replace it with the built-in SGD optimizer function

```python
# mnist_loss
lr = 1.
linear2 = nn.Linear(28*28,1)
opt = SGD(linear2.parameters(), lr)
train_model(linear2, 10, True)
```

    0.4932 0.4932 0.6807 0.874 0.9204 0.9355 0.9512 0.958 0.9638 0.9658 

```python
# softmax_loss
lr = 1e-2
linear3 = nn.Linear(28*28,2)
opt = SGD(linear3.parameters(), lr)
train_model(linear3, 10, False)
```

    0.5444 0.8984 0.9399 0.9575 0.9638 0.9648 0.9648 0.9653 0.9658 0.9663 

Instead of `train_model`, we can now transition over to using the built-in `Learner.fit` class method. Before we do this, lets redefine batch_accuracy for mnist and softmax separately as there is a need to follow the template here for this method

```python
def batch_accuracy_mnist(xb,yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```

```python
def batch_accuracy_ce(xb,yb):
    _,preds = torch.max(xb, 1)
    yb_squeezed = torch.squeeze(yb)
    #print(xb.shape, yb.shape, preds.shape, yb_squeezed.shape)
    correct = (preds == yb_squeezed)
    return correct.float().mean()
```

```python
learn_mnist = Learner(dls, nn.Linear(28*28,1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy_mnist)
```

```python
learn_ce = Learner(dls, nn.Linear(28*28,2), opt_func=SGD, loss_func=cross_entropy_loss, metrics=batch_accuracy_ce)
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
      <td>0.636874</td>
      <td>0.503316</td>
      <td>0.495584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.462960</td>
      <td>0.232174</td>
      <td>0.795388</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.172324</td>
      <td>0.162464</td>
      <td>0.853778</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.076681</td>
      <td>0.099876</td>
      <td>0.917566</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.041421</td>
      <td>0.074306</td>
      <td>0.935721</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.027634</td>
      <td>0.060111</td>
      <td>0.949951</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.021947</td>
      <td>0.051175</td>
      <td>0.956820</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.019395</td>
      <td>0.045197</td>
      <td>0.963690</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.018082</td>
      <td>0.040978</td>
      <td>0.966143</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.017282</td>
      <td>0.037855</td>
      <td>0.968597</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
learn_ce.fit(10, lr=1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy_ce</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.275928</td>
      <td>0.691266</td>
      <td>0.508832</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.218829</td>
      <td>0.307219</td>
      <td>0.883709</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.172936</td>
      <td>0.202616</td>
      <td>0.936703</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.144097</td>
      <td>0.161210</td>
      <td>0.954858</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.125797</td>
      <td>0.139614</td>
      <td>0.963199</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.113691</td>
      <td>0.126365</td>
      <td>0.965162</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.105253</td>
      <td>0.117359</td>
      <td>0.966143</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.099056</td>
      <td>0.110794</td>
      <td>0.965653</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.094285</td>
      <td>0.105764</td>
      <td>0.966634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.090465</td>
      <td>0.101761</td>
      <td>0.966143</td>
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
simple_net_ce = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,2) # 2 columns of activations from the final layer
)
```

```python
learn_mnist = Learner(dls, simple_net_mnist, opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy_mnist)
```

```python
learn_softmax = Learner(dls, simple_net_ce, opt_func=SGD, loss_func=cross_entropy_loss, metrics=batch_accuracy_ce)
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
      <td>0.361355</td>
      <td>0.396705</td>
      <td>0.512758</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.163877</td>
      <td>0.249748</td>
      <td>0.778214</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.088437</td>
      <td>0.121506</td>
      <td>0.909715</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.056420</td>
      <td>0.080100</td>
      <td>0.939647</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.041769</td>
      <td>0.061811</td>
      <td>0.956330</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.034433</td>
      <td>0.051752</td>
      <td>0.963199</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.030325</td>
      <td>0.045476</td>
      <td>0.965162</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.027719</td>
      <td>0.041226</td>
      <td>0.966634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.025876</td>
      <td>0.038150</td>
      <td>0.969578</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024465</td>
      <td>0.035804</td>
      <td>0.970559</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.023330</td>
      <td>0.033946</td>
      <td>0.972031</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.022387</td>
      <td>0.032427</td>
      <td>0.973013</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.021586</td>
      <td>0.031153</td>
      <td>0.974485</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.020896</td>
      <td>0.030063</td>
      <td>0.975466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.020291</td>
      <td>0.029114</td>
      <td>0.975466</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.019757</td>
      <td>0.028278</td>
      <td>0.976448</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.019280</td>
      <td>0.027533</td>
      <td>0.976938</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.018850</td>
      <td>0.026866</td>
      <td>0.977429</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.018460</td>
      <td>0.026265</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.018103</td>
      <td>0.025718</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
learn_ce.fit(20, 1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy_ce</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.085444</td>
      <td>0.098483</td>
      <td>0.967125</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.083706</td>
      <td>0.095735</td>
      <td>0.968597</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.081895</td>
      <td>0.093388</td>
      <td>0.969087</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.080118</td>
      <td>0.091353</td>
      <td>0.970069</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.078438</td>
      <td>0.089565</td>
      <td>0.970559</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.076881</td>
      <td>0.087976</td>
      <td>0.971050</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.075449</td>
      <td>0.086552</td>
      <td>0.971050</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.074135</td>
      <td>0.085264</td>
      <td>0.971050</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.072925</td>
      <td>0.084091</td>
      <td>0.972031</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.071808</td>
      <td>0.083017</td>
      <td>0.972031</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.070773</td>
      <td>0.082028</td>
      <td>0.972522</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.069809</td>
      <td>0.081112</td>
      <td>0.973013</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.068909</td>
      <td>0.080260</td>
      <td>0.973013</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.068065</td>
      <td>0.079465</td>
      <td>0.973503</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.067272</td>
      <td>0.078721</td>
      <td>0.973503</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.066524</td>
      <td>0.078021</td>
      <td>0.973503</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.065817</td>
      <td>0.077362</td>
      <td>0.973503</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.065148</td>
      <td>0.076739</td>
      <td>0.973994</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.064512</td>
      <td>0.076149</td>
      <td>0.973994</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.063907</td>
      <td>0.075588</td>
      <td>0.973994</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


### Pretty cool accuracy numbers there! :-)
