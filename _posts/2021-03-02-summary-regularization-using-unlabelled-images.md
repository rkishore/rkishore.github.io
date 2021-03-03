# Notes on "Universum Prescription: Regularization Using Unlabeled Data"



**Objective:** In this notebook, I note down my key takeaways from the paper ["Universum Prescription: Regularization Using Unlabeled Data"](https://arxiv.org/abs/1511.03719). This paper is of interest as I want to see if we can train classifiers that can reliably return "none-of-the-above" as a category for images that contain objects that are not part of the training set. In this paper, while the stated goal is different, i.e. they use these unlabeled images in the "none-of-the-above" class as a beneficial regularization technique to reduce overfitting, we still want to understand how exactly they did this and what we can learn from it towards my goal.

Previously, to achieve this goal of returning "none-of-the-above" as a category for images that don't contain the trained classes of interest, I had explored [using multi-label classification, and more specifically, the binary cross-entropy loss](https://rkishore.github.io/2021/02/23/pets-multi-label-classification.html) as recommended by Jeremy Howard in fastai coursework [fastai 2020 course, Lesson 6](https://course.fast.ai), the [textbook, Chapter 6](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527), [Lesson 10, part 2 2019 (watch till 52:44)](https://www.youtube.com/watch?t=2678&v=HR0lt1hlR6U&feature=youtu.be) and fastai forums[1](https://forums.fast.ai/t/lesson-9-discussion-wiki-2019/41969/513?u=rkishore),[2](https://forums.fast.ai/t/lesson-9-discussion-wiki-2019/41969/511?u=rkishore).

Broadly, this is all part of interest at work in incrementally training and improving classifiers in the field when we "miss" new objects and then incrementally add them to our model over time in an automated manner.

## Summary:

1. The authors augment their training data with large numbers of unlabeled images to see if it will help improve classification accuracy.

2. They make the following explicit assumptions:
    1. the proportion of unlabeled samples belonging to one of the labeled classes is negligible. Equivalently, the probability of a labeled sample existing in the unlabeled data is negligible
    2. Use of negative log-likelihood and/or cross-entropy loss as the loss function. They claim that the "loss function assumption is a detail rather than a limitation" in their Conclusion section. I assume this means we can use other loss functions? Not exactly sure what they mean but from what we have seen/heard before (see "Aside" later), this is not a good loss function for this problem.

3. Some implicit assumptions they make:
    1. labeled classes are typically small datasets and that in many practical cases, we only care about a small number of clases, either by problem design or due to high cost of labeling 
    2. It is easy to obtain a very large amount of unlabeled data

4.  They categorize these unlabelled images in three ways: (a) "dustbin" class, (b) "background" class and (c) "uniform prescription"

5. An attempt at explaining the three ways they categorize these unlabeled images during training is as follows:
    1. *Dustbin* class: create a separate class into which unlabeled images are classified under. Note that this approach has the best experimental results. Also note that in this case, the network has to learn parameters for this class as part of the training so it has to learn "negative features" compared to the labeled classes
    2. *Background* class: create a separate class into which unlabeled images are classified under. But here, no parameters to learn and some form a constant threshold is used. I did not fully understand this approach yet
    3. "Uniform prescription": from what I could understand, here, there is no additional class for unlabeled data. If an image from the unlabeled class shows up, then the labeled classes are all assigned a low probability of 1/k where k is the number of labeled classes. This seems more like some weird form of multi-label classification where if a "none-of-the-above" image shows up, the labeled classes will all have low activation values? Don't understand how this can work with Softmax and cross-entropy loss where the goal is to maximize the class probability of 1 class and minimize it for others. Regardless, the "dustbin" class separation has the best results.
    
6. The paper is split into a theoretical section and an experimental section. 
    1. In the theoretical section, the authors try to justify what they are doing using something called *Radamacher complexity*. I did not read this section carefully. I was more interested in what the experiments show.
    2. In the experimental section, they show results from their three kinds of categorization for the unlabelled images for the CIFAR10, CIFAR100 and Imagenet datasets. They also use an STL-10 dataset that I don't know much about.
        1. They use their own 21-layer and 17-layer CNNs as the backbone for classification. The 21-layer CNN is used for CIFAR10 and CIFAR100 while the 16-layer CNN is used for Imagenet.
        2. All pooling layers are max-pooling and ReLUs are used as the non-linearity. These backbones are similar to VGG19. No residual layers yet.
        3. For CIFAR10/CIFAR100, they use 32x32 input images and for Imagenet, they 64x64 input images. SGD with momentum of 0.9 and a minibatch size of 32. The initial learning rate is 0.005 which is halved every 60000 minibatch iterations for CIFAR10/CIFAR100 and every 600k iterations for Imagenet. Training stops at 400k iterations for CIFAR10/CIFAR100 and at 2.5M iterations for Imagenet. Wow! No transfer learning for sure and no learning rate optimizations like CLR. They use data augmentations (horizontal flip, scale, crop) during training. 
        4. During training, an unlabeled sample is randomly presented with a probability p (0.2 used in experiments)
        5. Results show that test error rate is comparable to best of breed work around 2015 (pre-ResNets) for CIFAR10 and CIFAR100. They don't comment about how their Imagenet results compare to state-of-the-art at that time and are in general, light on the details.
        6. For all datasets, the "dustbin" class approach yields the best results. They explain this by saying that since this class has trained parameters, it makes it adapt best to unlabelled samples. Don't think this is sufficient enough an explanation.
        7. For STL-10, their results for this dataset show that a LOT of unlabelled data is needed for this to work + the unlabelled data should not have labeled class data inside or vice-versa. This is a key result that affects the practicality of using this approach.
        8. Another result is that the regularization provided by using these unlabelled samples is comparable to that provided by using *dropout*.

7. **Key takeaways**: 
    1. You need a LOT of data in the unlabeled class (tens of millions of images that DON'T contain images from the labeled classes) to pull this off well. This requirement seems impractical. How do we curate such a dataset even if Google/Bing/DuckDuckGo image search can get the images? 
    2. The authors don't explicitly perform the experiment where images outside the labeled classes are inferred successfully with a "none-of-the-above" classification. This is something we may have to do on our own time. Along the same lines, what happens if there are more than one class of labeled objects in the input image?

### Non-technical comments:

1. Figure 1 is extremely hard to see. Wish the first author would have taken the time to blow it up and explain it slightly better. Are page limits the reason why so much data is condensed in such a tiny space with such a small font in this figure? 

2. Relating to Figure 1, and in the section on experimental results for Imagenet, the authors don't tell us the amount of unlabeled images they used and this is a critical point that is missing as the amount of unlabeled data for this technique to work for CIFAR10, CIFAR100 and STL-10 is *80 million images*! In general, the results and explanation for Imagenet leaves a lot to be desired (like how it compares to state-of-the-art)?


## Aside (Jeremy's view)

In [fastai coursework (watch till 52:44)](https://www.youtube.com/watch?t=2678&v=HR0lt1hlR6U&feature=youtu.be), Jeremy actually says, 
1. "most of the time, we don't want Softmax"! 
2. And "why do we use it? because we grew up with Imagenet and Imagenet was specifically curated so that it only has ONE of those classes in it and always has one of those classes in it"! 
3. He goes on to say how famous academic researchers make the mistake of trying to use Softmax with a dustbin class like this paper does and he does not think its a good idea as its very hard to hack/train a neural network to recognize the negative of ALL the classes its training on to begin with.

This view is important to me as Jeremy is probably the one guy who is teaching us how to build practical, easy to train models and has the breadth of experience that shows in fastai repeatedly. He also says that we can try this out with and without Softmax and we will likely get a better result without Softmax. His gut/intuition appeals to my experimental background in general.
