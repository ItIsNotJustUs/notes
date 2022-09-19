# 常见损失函数

## 1、误差

真实值 与 模型的预测值 误差必然存在	$\rightarrow$	$真实值-预测值=\varepsilon $ 

那么对于每个样本就有： 
$$
\begin{array}{}
y^{(i)} & =  & \theta ^T x^{(i)} & + & \varepsilon ^{(i)} \\
真实值 & & 预测值 & & 误差
\end{array}
$$
一般假设：误差$\varepsilon ^{(i)}$独立同分布，并且均服从均值为$0$,方差为$\theta^2$的高斯分布(正态分布,)。

## 2、损失函数

以线性回归为例：

一般情况误差有$\varepsilon^{(i)}$ 正有负，如果直接对$\varepsilon^{(i)}$求和，则会出现正负抵消的情况，反映不出整体误差情况。如果使用平方和，不仅可以避免正负抵消的缺陷，而且整体的误差分布不变，所以一般使用平方和做损失函数。
$$
\sum_{i=1}^{m}(y_i-\hat{y}_i )^2 = \sum_{i=1}^{m}(y_i-X_iw)^2
$$
其中$y_i$是样本$i$对应的真实值，$\hat{y}_i$（即$X_iw$）是样本$i$在一组参数$w$下的预测值。

**极大似然估计解释损失函数**

由于假设 $\varepsilon^{(i)} \sim N(0, \sigma^2)$，则
$$
p(\varepsilon^{(i)})=\frac{1}{\sqrt{2\pi} \sigma} \exp(-\frac{(\varepsilon^{(i)})^2}{2\sigma^2}) \tag{1}
$$
预测值与误差：
$$
y^{(i)} = \theta^Tx^{(i)}+\varepsilon^{(i)} \tag{2}
$$
将$(2)$带入$(1)$中：
$$
p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
$$
从$(2)$式可以看出，**误差**$\varepsilon^{(i)}$**越小，**$P(y^{(i)}|x^{(i)};\theta)$**概率越大，说明预测值与真实值越接近。**

因线性回归模型是一条直线（或超平面）拟合多个点，所以需要满足所有误差取得最小值，即所有概率的乘积最大化，符合**似然函数**：
$$
L(\theta)=\prod_{i=1}^{m}  p(y^{(i)}|x^{(i)};\theta) = \prod_{i=1}^{m}\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)} - \theta^Tx^{(i)} )^2}{2\sigma^2})
$$
对数似然：$\log L(\theta) = \log \prod\limits_{i=1}^{m}\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)} - \theta^Tx^{(i)} )^2}{2\sigma^2})$

化简：
$$
\begin{array}{}
\sum\limits_{i=1}^{m} \log \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)} - \theta^Tx^{(i)} )^2}{2\sigma^2}) \\ 
= m \log \frac{1}{\sqrt{2\pi}\sigma} -\frac{1}{\sigma^2} \cdot \frac12 \sum\limits_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2
\end{array}
$$
 为了使得似然整体变大，$ \frac12 \sum\limits_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2$ 自然越小越好。

也就得到目标损失函数$Loss(\theta) =  \frac12 \sum\limits_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2$

## 3、常见损失函数

损失函数的选择会影响模型最终的收敛效果(甚至能否收敛)，也会让我们选择不同的优化器来求解。

我们对误差($\varepsilon$)不同的衡量(上面假设$\varepsilon^{(i)} \sim N(0, \sigma^2)$)就会产生各种各样的损失函数。

> 摘抄[常见的损失函数(loss function)总结]([常见的损失函数(loss function)总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/58883095))

### 3.1. 0-1损失函数

解释：指预测值和目标值不相等为1， 否则为0:

公式:$ L(Y,f(X)) = \left \{  \begin{array}{}  1, &Y \ne f(X) \\ 0, & Y=f(X) \end{array} \right.$

特点：

- 0-1损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用.
- **感知机**就是用的这种损失函数。但是相等这个条件太过严格，因此可以放宽条件，即满足 $|Y−f(X)|<T $时认为相等，$ L(Y,f(X)) = \left \{  \begin{array}{}  1, &|Y - f(X)| \ge T \\ 0, & |Y - f(X)| <T \end{array} \right.$

### 3.2. **绝对值损失函数**

解释：绝对值损失函数是计算预测值与目标值的差的绝对值：

公式：$L(Y,f(X))=|Y-f(X)|$

### 3.3.  log对数损失函数

公式：$L(Y, P(Y|X)) = -\ log P(Y|X)$

特点：

- log对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，如果需要知道结果属于每个类别的置信度，那它非常适合。
- 健壮性不强，相比于hinge loss对噪声更敏感。
- **逻辑回归**的损失函数就是log对数损失函数。

### 3.4. 平方损失函数

公式： $L(Y|f(X))=\sum\limits_N(Y-f(X))^2$

特点：经常应用于回归问题

补充：

1. 和方差(The sum of squares due to error, SSE):	$SSE = \sum i = \sum\limits_{i} w_i(y_i-\hat{y_i})^2$
2. 均方方差(Mean squared error, MSE): 	$MSE = \frac{SSE}{n} = \frac1n \sum\limits_{i} w_i (y_i-\hat{y_i})^2$
3. 均方根(标准差,Root mean squared error, RMSE):		$RMSE = \sqrt{MSE}=\sqrt{\sum\limits_{i}w_i (y_i-\hat{y_i})^2}$

### 3.5. 指数损失函数（exponential loss）

公式: $L(Y|f(X)) = \exp[-yf(x)]$

特点：

- 对离群点、噪声非常敏感。经常用在AdaBoost算法中。

### 3.6. Hinge 损失函数

公式: $L(y,f(x)) = max(0, 1-yf(x))$

特点：

- $hinge$损失函数表示如果被分类正确，损失为$0$，否则损失就为 $1−yf(x)$ 。**SVM**就是使用这个损失函数。
- 一般的 $f(x)$ 是预测值，在$-1$到$1$之间，$ y$ 是目标值($-1$或$1$)。其含义是， $f(x)$ 的值在$-1$和$+1$之间就可以了，并不鼓励$ |f(x)|>1$ ，即并不鼓励分类器过度自信，让某个正确分类的样本距离分割线超过$1$并不会有任何奖励，从而**使分类器可以更专注于整体的误差。**

(3) 健壮性相对较高，对异常点、噪声不敏感，但它没太好的概率解释。

### 3.7. 感知损失(perceptron loss)函数

公式: $L(y,f(x)) = max(0,-f(x))$

特点：

- 是Hinge损失函数的一个变种，Hinge loss对判定边界附近的点(正确端)惩罚力度很高。而perceptron loss**只要样本的判定类别正确的话，它就满意，不管其判定边界的距离**。它比Hinge loss简单，因为不是max-margin boundary，所以模**型的泛化能力没 hinge loss强**。

### 3.8. **交叉熵损失函数 (Cross-entropy loss function)**

公式$L(y,\hat{y}) = -\frac1n \sum\limits_i [y_i \ln \hat{\hat{y_i}} + (1-y)\ln (1-\hat{y_i})]$

特点：

- 本质上也是一种**对数似然函数**，可用于二分类和多分类任务中。
- 二分类问题中的$loss$函数（输入数据是$softmax$或者$sigmoid$函数的输出）：

$$
loss=−\ln∑x[y \ln \hat{y}+(1−y)ln⁡(1−\hat{y})]
$$

多分类问题中的$loss$函数（输入数据是$softmax$或者$sigmoid$函数的输出）：
$$
loss=−\ln \sum\limits_i y_iln\hat{y_i}
$$

- 当使用sigmoid作为激活函数的时候，常用**交叉熵损失函数**而不用**均方误差损失函数**，因为它可以**完美解决平方损失函数权重更新过慢**的问题，具有“误差大的时候，权重更新快；误差小的时候，权重更新慢”的良好性质。






​