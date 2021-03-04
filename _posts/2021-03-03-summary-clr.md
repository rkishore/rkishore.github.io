# Notes on "Cyclical Learning Rates for Training Neural Networks"



**Objective:** In this notebook, I review [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf). This paper is very relevant as it helps train networks much faster while also solving the hard practical problem of finding the optimum learning rates to train with, for improved accuracy. 

I learnt about this paper as part of the [fastai 2020 course](https://course.fast.ai), and the [textbook, Chapter 5](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527). `One-cycle` training is default in the [fastai library](https://docs.fast.ai). And the first recommended step before training any model is to use the [`learning rate finder`](https://fastai1.fast.ai/callbacks.lr_finder.html) to find the best learning rates to use. The basis for both these defaults is this paper so its a good idea to understand the details well. While fastai implementations differ from the exact approach proposed in this paper (we will cover the differences), they are fundamentally using the principles from this paper.

## Summary:

1. Motivation:
    1. Training a deep neural network is a difficult global optimization problem.
    2. Weights in a deep neural net are typically learned using SGD. `Wt = Wt-1 - Et * dL/DW`, where `L` is the loss function, `Et` is the learning rate and `Wt` is the weight at time/iteration `t`. 
    3. Learning rate is the most important hyper-parameter to tune for training deep neural nets.
    4. We need a better approach to tune the learning rate compared to prior heuristic approaches (breaking up the total epochs into 3-4 large chunks. Keeping the LR fixed in each chunk and monotonically stepping it **down** when going from one chunk to another).

2. Proposal and key observations:

    1. During training, cyclically vary the learning rate between bounds `base_lr` and `max_lr` where `base_lr` < `max_lr` (see plot from [1](https://arxiv.org/pdf/1506.01186.pdf)).
    
    <img src="/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/cyclical-lr.png" alt="Cyclical LR" width="400"/>
    
    2. To determine the `base_lr` and `max_lr` bounds (called the **LR range test** in the paper),
        1. Start at a very low learning rate and increase it linearly for several epochs between low and high LR values as you train
        2. Plot the accuracy vs learning rate
        3. Note the LR where the accuracy starts to increase (for the `base_lr`) and where the accuracy starts to become ragged or starts decreasing (for the `max_lr`). These two LR values are good choices for `base_lr` and `max_lr` respectively. Alternately, use the rule of thumb that the optimum LR is usually within a factor of two of the largest one that converges. So `base_lr` can be set to `1/3` or `1/4` of `max_lr`.
        4. See plot from paper below that shows how the base_lr and max_lr are selected for CIFAR10 
        
        <img src="/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/lr-range-test-cifar10.png" alt="Cyclical LR" width="400"/>
        
    3. The key motivation behind this cyclical lr policy is the observation that increasing the learning rate might have a short term negative effect and yet achieve a longer term beneficial effect. This leads to the idea of letting the learning rate vary within a range of values rather than adopting a stepwise fixed or exponentially decreasing value. 
    
    4. The intuition behind why increasing the lr helps comes from considering the loss function topology and how the difficulty in minimizing the loss comes from traversing the saddle points. Saddle points have small gradients that slow the learning process. Increasing the learning rate allows more rapid traversal of the saddle point plateuaus. 
    
    5. Note that using the **LR range test**, it is likely that the optimum learning rate will be between the bounds and near optimal learning rates will be used throughout training. 
    
    6. The implementation of the cyclical learning rate policy is really simple. The easiest way to understand this is using code as described in the paper.
    

```python
import math
from matplotlib import pyplot as plt

'''
As an example, let's take CIFAR-10 as the dataset 
on which we use this policy on.

CIFAR-10 has 50,000 training images and the batchsize 
is 100 so an epoch is 50,000/100 = 500 iterations.

epochCounter is just the training iteration going on.
In the code below, we just set it to discrete values 
to see the behavior of the 
learning rate vs epochCounter/iteration.
'''
epochCounters = [i for i in range(0, 12500, 500)]

'''
As the name suggests, cyclical learning rate works 
in cycles. The learning rate increases and decreases
in each cycle. Cycle length is in iterations and so is 
stepsize, which is half the cycle length. 

The learning rate increases from `base_lr` to `max_lr` 
from the beginning of the cycle to `stepsize`. Beyond 
stepsize iterations, it will decrease from `max_lr` to 
`base_lr` till the end of the cycle.

For example, if it's the first cycle, then the 
starting iteration number is 0. The learning rate 
will increase from `base_lr` till `max_lr` till 
`stepsize` iterations beyond which it will reduce 
from `max_lr` to `base_lr` till the end of the cycle.

The paper tells us that it is good to set the `stepsize`
to 2-10 times the number of iterations in an epoch. 
Here, we use stepsize = 4x500 = 2000 as is done in the paper
for CIFAR-10. The length of each cycle is then 
2*2000 = 4000 iterations. 
'''
stepsize = 2000

'''
We assume that using the LR-range test, these base_lr 
and max_lr values are found. We use them here as our 
bounds for the cyclical training
'''
base_lr = 0.001
max_lr = 0.006
```

```python
'''
The function below implements the cyclical learning rate 
policy. Note that here we just print out the learning rate
and demonstrate its cyclical nature.

During training, depending on the training iteration going on,
the learning rate will be set as per the code below.
'''
def print_stats_for_clr(epochCounters):
    cyc_lr = []
    for epochCounter in epochCounters:
        '''
        The three lines below are the exact code implementation
        as specified by the paper in section 3.1
        '''
        cycle = math.floor(1 + epochCounter/(2*stepsize))
        x = abs(epochCounter/stepsize - 2*cycle + 1)
        lr = opt_lr + (max_lr - opt_lr) * max(0, (1-x))
        #print(epochCounter, cycle, x, lr)
        cyc_lr.append(lr)
    return cyc_lr
cyc_lr_vals = print_stats_for_clr(epochCounters)
#print(cyc_lr_vals)

plt.plot(epochCounters, cyc_lr_vals)
plt.xlabel("Cycle #")
plt.ylabel("Iteration #")
```




    Text(0, 0.5, 'Iteration #')




![png](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/output_3_1.png)


### Results: how does this policy do?

The authors evaluate this cyclical learning rate policy on 
- Different datasets such as CIFAR-10, CIFAR-100 and Imagenet. 
- Different network architectures: for CIFAR-10 and CIFAR-100, one backbone network they use is the "CIFAR-10 architecture and hyper-parameter settings on the Caffe website". We have to find what this backbone is. They also use other backbones like Resnets, Stochastic depth networks and densenets. For Imagenet, they use both Alexnet and Googlenet as the backbone.
- CIFAR-10 with different adaptive learning rate algorithms like ADAM, Nesterov, RMSProp, AdaGrad, AdaDelta
- CIFAR-10 with sigmoid non-linearity (instead of ReLU) and batch normalization

Overall, **we see that in most, if not all cases, the accuracy with cyclical learning rates is close to, the same as or higher than what approaches without cyclical LR policies get. In cases involving CIFAR-10, these accuracy values are arrived at a lot sooner. With Imagenet, the same number of iterations are needed as policies without cyclical learning rates but the accuracies achieved by cyclical learning rates are marginally higher (0.4% and 1.4% higher for Alexnet and Googlenet).**

Key result plots from the paper are shown next along with my notes. Lots of good experimental data and a lot of good work by Leslie Smith, the author.

CLR Result             |  CLR Result 
:-------------------------:|:-------------------------:
![CLR-Figure1](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-cifar10-fig1.png) | ![CLR-Figure5](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-cifar10-fig5.png)
![CLR-Table1](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-table1.png) | ![CLR-Table4](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-table4.png)

LR Range CIFAR10      |  LR Range Imagenet+Alexnet    | LR Range Imagenet+Googlenet
:-------------------------:|:-------------------------:|:-------------------------:
![LR-range-test-CIFAR10](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/lr-range-test-cifar10.png) | ![LR-range-Imagenet-Alexnet](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-alexnet-lr-range-test-fig7.png) | ![LR-range-Imagenet-Googlenet](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-googlenet-fig11.png)

![Imagenet-Alexnet accuracy](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-alexnet-fig9-10.png)
![Imagenet-Googlenet accuracy](/images/Paper_summary_cyclical_learning_rates_for_neural_nets_2017_files/clr-googlenet-fig12-13.png)

### Modified implementation in fastai when it comes to the lr_range test

In fastai, I see that the implementation of the lr range test (called *lr_find* in fastai) to determine the base_lr and max_lr bounds is [slightly different](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate). In fastai, the *loss* is plotted with respect to increasing the learning rate and we don't even need a whole epoch to find the optimum learning rate bounds as we change the learning rate across minibatches in an epoch and note down the loss. 

Using the loss instead of the accuracy seems to be a better way to go as loss and learning rate are more directly related than learning rate and accuracy. Remember that during SGD, `Wt = Wt-1 - Et * dL/DW`, where `L` is the loss function, `Et` is the learning rate and `Wt` is the weight at time/iteration `t`. Of course, using loss instead of accuracy means that we need to deal with the sensitivity of the loss function. We need a way to smoothen it out when we plot it so as to make it easy to infer from. See details of the exponentially weighted smoothing that is carried out in this case as discussed in this [reference](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate).
