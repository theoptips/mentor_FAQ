# FAQ for Udacity Intro to Machine Learning Nanodegree | Mentorship FAQ

# FAQ
- FAQ 001	Error VGG object has no attribute error. Ans: 001, 002
- FAQ 002	How to access hidden layers of Pytorch pre-trained model Ans: 001, 002
- FAQ 003 	How to access a Pytorch Sequential object Ans: 001, 002
- FAQ 004	Google Colab Best Practice Ans: 003,
- FAQ 005	Article recommendations for imbalanced data classes Ans: 004, 005, 009
- FAQ 006	Transfer learning with Pytorch Ans: 006
- FAQ 007 	Categorical Encoding Ans: 007
- FAQ 008	How to check the rank of a tensor? Ans: 010
- FAQ 009	Anaconda environment cheat sheet Ans: 011
- FAQ 010	Pytorch Cheatsheet | Pytorch 101 Ans: 012, 006
- FAQ 011	Trending deep learning technology GANs: living portrait aka few-shot adversarial learning 015 014, AI portraits 013
- FAQ 012   VGG16... multiple linear layers within the classification layer, so to get the in_features, the command would be model.classifier[0].in_features ... resnet18, the command would need to be model.fc.in_features.  How can I write a general command, which would give me the in_features, no matter which model is passed? asked by FK. This question is also interpreted as building transfer learning pipeline, a workflow, aka a pre-trained model workflow in pytorch. 

# ANSWERS
- Answer 001 [Pytorch Forum VGG Object Has No Attribute Error](https://discuss.pytorch.org/t/vgg-object-has-no-attribute-fc/9124/3)
- Answer 002 [Pytorch Transfer learning on Medium](http://bit.ly/transfer_learning_pytorch)
- Answer 003 When starting a session in colab, the runtime environment is reset. Be sure to save checkpoint and data, back up the Jupyter Notebook often. 
- Answer 004 [Techniques handling imbalanced data](https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html)
- Answer 005 [What metrics should we use on imbalanced dataset](https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba)
- Answer 006 [Transfer learning with pytorch code snippets Uniqtech on Medium](https://medium.com/data-science-bootcamp/transfer-learning-with-pytorch-code-snippet-load-a-pretrained-model-900374950004)
- Answer 007 [Categorical encoding](https://pbpython.com/categorical-encoding.html)
- Answer 008 [One Hot Encode Data in Machine Learning Machine Learning Mastery](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
- Answer 009 [Imbalanced Datasets](https://blog.dominodatalab.com/imbalanced-datasets/)
- Answer 010 Check the rank of a tensor by checking the len of its shape tuple `len(tensor.shape)` For example if the rank is 3 then it is a 3-dimensional tensor.
- Answer 011 [Anaconda Miniconda Cheatsheet for Data Scientists](https://link.medium.com/Rw63GQ2peY)
- Answer 012 [Pytorch Cheatsheet for Beginners | Udacity Deep Learning Nanodegree | Udacity Intro to Machine Learning Uniqtech on Medium](https://medium.com/@uniqtech/pytorch-cheat-sheet-for-beginners-and-udacity-deep-learning-nanodegree-5aadc827de82)
- Answer 013 [Create your own Renaissance portraits](https://www.fastcompany.com/90376689/what-you-look-like-as-an-renaissance-painting-according-to-ai)
- Answer 014 [Living portraits by Samsung](https://petapixel.com/2019/05/24/samsung-ai-can-turn-a-single-portrait-into-a-realistic-talking-head/)
- Answer 015 “Few-Shot Adversarial Learning of Realistic Neural Talking Head Models,” a team of researchers at the Samsung AI Center in Moscow, Russia
- Answer 016 Use `model.children()` to access the layers of a NN in Pytorch. It is a generator, so we `append` each layer into a list variable called `result`, access and replace the entire last section using `result[-1]`. `type()` check the last element, if `Sequential` then use `result[-1][0].in_features`, else if `type` is `Linear` use `result[-1].in_features`

# Related FAQ 
- Pytorch VGG error no attribute 001 002
- Pytorch general 012 006
- Project CharityML 

