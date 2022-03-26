## Logistic Regression

Hello guys, This is Nikhil Nandam, back again with another article. This article is going to deal with Logistic Regression. So let's begin!
<hr>


### Logistic Regression

Logistic Regression is a supervised machine learning model. Interestingly, Logistic Regression is not a Regression method as the name suggests but it is actually a **classification model** in machine learning.

Since Logistic Regression is a classification model, the target values of the model are binary values.

Logistic Regression is an extension of Linear Regression and that extension is in applying the *logistic function* to the predicted target value obtained using Linear Regression to obtain the target value in Logistic Regression.

<hr>

### Logistic Function (Sigmoid Function)

The logistic function transforms real-valued input to an output number <img src="https://latex.codecogs.com/svg.image?\inline&space;y" title="https://latex.codecogs.com/svg.image?\inline y" />
between 0 and 1, interpreted as the probability the object belongs to the positive class, given it's input features <img src="https://latex.codecogs.com/svg.image?\inline&space;(x_0,&space;x_1,&space;x_2,&space;...&space;,&space;x_n)" title="https://latex.codecogs.com/svg.image?\inline (x_0, x_1, x_2, ... , x_n)" />.

The target values in Logistic Regression are obtained using the formula below.

<img src="https://latex.codecogs.com/svg.image?&space;\hat{y}&space;=&space;logistic(x_1\hat{w_1}&space;&plus;&space;x_2\hat{w_2}&space;&plus;&space;x_3\hat{w_3}&space;&plus;&space;...&space;&plus;&space;x_n\hat{w_n}&space;&plus;&space;\hat{b})" title="https://latex.codecogs.com/svg.image?\hat{y} = logistic(x_1\hat{w_1} + x_2\hat{w_2} + x_3\hat{w_3} + ... + x_n\hat{w_n} + \hat{b})" />

The logistic function here is the sigmoid function represented by <img src="https://latex.codecogs.com/svg.image?\inline&space;S(x)" title="https://latex.codecogs.com/svg.image?\inline S(x)" />
whose formula and graph are shown below.

<img src="https://latex.codecogs.com/svg.image?&space;S(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}}" title="https://latex.codecogs.com/svg.image?S(x) = \frac{1}{1 + e^{-x}}" />

![1200px-Logistic-curve.svg.png](https://cdn.hashnode.com/res/hashnode/image/upload/v1635867677456/UxiA5GRdw.png)

So now, we can more precisely formulate <img src="https://latex.codecogs.com/svg.image?\inline&space;\hat{y}" title="https://latex.codecogs.com/svg.image?\inline \hat{y}" />.

<img src="https://latex.codecogs.com/svg.image?&space;\hat{y}&space;=&space;\frac{1}{1&space;&plus;&space;\exp{[-(x_1\hat{w_1}&space;&plus;&space;x_2\hat{w_2}&space;&plus;&space;x_3\hat{w_3}&space;&plus;&space;...&space;&plus;&space;x_n\hat{w_n}&space;&plus;&space;\hat{b})]}}" title="https://latex.codecogs.com/svg.image?\hat{y} = \frac{1}{1 + \exp{[-(x_1\hat{w_1} + x_2\hat{w_2} + x_3\hat{w_3} + ... + x_n\hat{w_n} + \hat{b})]}}" />

As we can observe from the graph of the sigmoid function <img src="https://latex.codecogs.com/svg.image?\inline&space;S(x)" title="https://latex.codecogs.com/svg.image?\inline S(x)" />
, the <img src="https://latex.codecogs.com/svg.image?\inline&space;\hat{y}" title="https://latex.codecogs.com/svg.image?\inline \hat{y}" />
values ranges between 0 and 1. So therefore, 

<img src="https://latex.codecogs.com/svg.image?&space;\hat{y}&space;\in&space;(0,&space;1)" title="https://latex.codecogs.com/svg.image?\hat{y} \in (0, 1)" />

<hr>

### Interpreting Output

The output from the Logistic Regression formula can be interpreted as the probability of the input data instance belonging to the positive class.

- A value highly close to 1, indicates a high probability of the input data instance belonging to the positive class.

- A value highly close to 0, indicates a high probability of the input data instance belonging to the negative class.

#### Thresholds

- An input data instance, whose output value from the logistic function lies between 0.5 and 1, that is, 
<img src="https://latex.codecogs.com/svg.image?\inline&space;\hat{y}&space;\geq&space;0.5" title="https://latex.codecogs.com/svg.image?\inline \hat{y} \geq 0.5" />
is classified as belonging to the positive class.

Similarly,

- An input data instance, whose output value from the logistic function lies between 0 and 0.5, that is, 
<img src="https://latex.codecogs.com/svg.image?\inline&space;\hat{y}&space;<&space;0.5" title="https://latex.codecogs.com/svg.image?\inline \hat{y} < 0.5" />
is classified as belonging to the negative class.

<hr>

### Sci-Kit Learn Module Implementation

The following code demonstrates use of Sci-Kit Learn's LogisticRegression class to perform classification.

```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
```

The accuracy of the model on the training set and test set and calculated using the code below.

```
print('Accuracy of Logistic regression classifier 
      on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier 
      on test set: {:.2f}'.format(clf.score(X_test, y_test)))
```

<hr>

### Advantages and Disadvantages

#### Advantages

- Logistic regression is easier to implement, interpret, and very efficient to train.

- It can easily extend to multiple classes(multinomial regression) and a natural probabilistic view of class predictions.

- It can interpret model coefficients as indicators of feature importance.


#### Disadvantages

- The major limitation of Logistic Regression is the assumption of linearity between the dependent variable and the independent variables.

- Non-linear problems can’t be solved with logistic regression because it has a linear decision surface. Linearly separable data is rarely found in real-world scenarios.

- It is tough to obtain complex relationships using logistic regression.

<hr>

That's a wrap guys! Hope you found this article insightful. Please do leave a like and comment down below any suggestions. Have a wonderful day!!! 😀

![thank-you-2.jpg](https://cdn.hashnode.com/res/hashnode/image/upload/v1636382672991/F_KAideLd.jpeg)







