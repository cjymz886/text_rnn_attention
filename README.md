# Text classification with CNN and Word2vec
本文是继自己上的blog“text-cnn”后，基于同样的数据集，嵌入词级别所做的RNN+ATTENTION模型所做的文本分类实验结果；<br><br>
本实验的主要目是为了探究在同样的数据情况，CNN模型与RNN+attention模型训练的效果对比，训练结果显示在验证集上CNN为96.5%，RNN+attention为96.8%；<br><br>
有兴趣可以阅读我的：[text-cnn](https://github.com/cjymz886/text-cnn)<br><br>

1 环境
=
python3<br>
tensorflow 1.3以上CPU环境下<br>
gensim<br>
jieba<br>
scipy<br>
numpy<br>
scikit-learn<br>

2 CNN卷积神经网络
=
模型RNN+ATTENTION配置的参数在text_model.py中，具体为：<br><br>
![image](https://github.com/cjymz886/text_rnn_attention/blob/master/images/config_rnn.png)<br><br>
模型RNN+ATTENTION大致结构为：<br><br>
![image](https://github.com/cjymz886/text-cnn/blob/master/images/text_cnn.png)

3 数据集
=
本实验同样是使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议;<br><br>
文本类别涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；<br><br>
cnews.train.txt: 训练集(5000*10)<br>

cnews.val.txt: 验证集(500*10)<br>

cnews.test.txt: 测试集(1000*10)<br><br>

训练所用的数据，以及训练好的词向量可以下载：链接: [https://pan.baidu.com/s/1gka7SgYIRijSaXgRfYZzwA ](https://pan.baidu.com/s/1gka7SgYIRijSaXgRfYZzwA)，密码: mmbk<br><br>

4 预处理
=
本实验主要对训练文本进行分词处理，一来要分词训练词向量，二来输入模型的以词向量的形式；<br><br>
另外，除掉文本的标点符号，也使用./data/stopwords.txt文件进行停用词过滤;<br><br>
处理的程序都放在loader.py文件中；<br><br>


5 运行步骤
=
python train_word2vec.py，对训练数据进行分词，利用Word2vec训练词向量(vector_word.txt)<br><br>
python text_train.py，进行训练模型<br><br>
python text_test.py，对模型进行测试<br><br>
python text_predict.py，提供模型的预测<br><br>


6 训练结果
=
运行：python text_train.py<br><br>
本实验经过6轮的迭代，满足终止条件结束，在global_step=3200时在验证集得到最佳效果96.5%<br><br>
![image](https://github.com/cjymz886/text_rnn_attention/blob/master/images/train_rnn.png)

7 测试结果
=
运行：python text_test.py<br><br>
对测试数据集显示，test_loss=0.13，test_accuracy=95.8%，其中“体育”类测试为100%，整体的precision=recall=F1=96%<br><br>
![image](https://github.com/cjymz886/text_rnn_attention/blob/master/images/test_rnn.png)

8 预测结果
=
运行:python text_predict.py <br><br>
随机从测试数据中挑选了五个样本，输出原文本和它的原文本标签和预测的标签，下图中5个样本预测的都是对的；<br><br>
![image](https://github.com/cjymz886/text_rnn_attention/blob/master/images/predict_rnn.png)

9 对比结论
=
在与cnn模型对比中发现，训练中在验证集上准确率96.8%是略优于cnn的，但是在测试集上，并没有cnn模型表现的好；我推测的其中原因是，CNN处理文本的长度为600，而RNN+ATTION处理的文本长度为200，而后者也不能处理太长的文本，文本越长，包含的特征信息越多，所以从整体上来看，我个人觉得CNN模型更适合长文本的分类任务；<br><br>

9 参考
=
1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2. [gaussic/text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
3. [YCG09/tf-text-classification](https://github.com/YCG09/tf-text-classification)
