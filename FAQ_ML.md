# FAQ for Udacity Advanced achine Learning Nanodegree | Mentorship FAQ

# FAQ
- FAQ 000	When in doubt email support@udacity.com Ans: 000
- FAQ 001	Amazon SageMaker 001
- FAQ 002	Intermediate Python 002
- FAQ 003	Intermediate to Advanced Amazon SageMaker 003
- FAQ 004	Trending deep learning technology GANs: Ans: intro to ML FAQ 011
- FAQ 005	How to build a transfer learning model pipeline / workflow in Pytorch Intro to ML FAQ 012
- FAQ 006	Github contribution guideline, how do you make your commits count, how to get credit for the github activity tracker Ans: 004
- FAQ 007	Reproducible research, deep learning, machine learning Ans: 006
- FAQ 008	What should I expect at the end of the course? What would I gain from this course? Ans: 007
- FAQ 009 	Data Augmentation in Computer Vision, why and how to do it Ans 008
- FAQ 010 	Learn more about reinforcement learning. Reinforcement Learning resources 009
- FAQ 011	What should I know before starting the capstone project? Answer 010 Also see FAQ 015
- FAQ 012 Machine learning textbook Ans 011
- FAQ 013 Capstone suggested format Ans 010
- FAQ 014 What is the difference between intro to machine learning, machine learning engineer, deep learning, artificial intelligence, self-driving, computer vision nanodegrees Ans 012
- FAQ 015 Important things to know before capstone! Ans 013
- FAQ 016 Where to find my AWS promotional credit? Included AWS credit for GPU training Ans 014
- FAQ 017 Tell me more about benchmarking. Answer 015
- FAQ 018 Cannot find ml.p2.xlarge in limit request Access to GPU. "on the aws console, when i search for p2.xlarge, there's just no result at all." Ans 16
- FAQ 019 ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateEndpoint operation Ans 17 Ans 19 Ans 20
- FAQ 020 What are the differences of aws sagemaker instance types. What is `ml.p2.xlarge` Ans 18
- FAQ 021 How do I get started on my capstone project? Ans 21
- FAQ 022 pre-requisites Ans 22
- FAQ 023 Difference between the intro to ML and MLEND the two Machine Learning Engineering Nanodegrees Ans 23
- FAQ 024 Help! My model performance isn't improving the loss is not decreasing. Ans 24
- FAQ 025 AWS SageMaker XGBoost error `tree pruning end, 1 roots, 0 extra nodes, 0 pruned nodes, max_depth=0 [1]#011train-error:0#011validation-error:0` `Will train until validation-error hasn't improved in 10 rounds.
[02:58:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 0 extra nodes, 0 pruned nodes, max_depth=0` You can tell that the XGBoost tree never trained because the train error and validation error are both zero. Ans 25


# ANSWERS
- Answer 000 When in doubt email support@udacity.com
- Answer 001 [Amazon SageMaker Getting Started | Deep Learning Production Deployment on Amazon Web Services by Uniqtech on Medium](https://medium.com/swlh/jupyter-notebook-on-amazon-sagemaker-getting-started-55489f500439)
- Answer 002 [Real Python Writing Pythonic Code](https://realpython.com/learning-paths/writing-pythonic-code/)
- Answer 003 [Amazon SageMaker Now Comes with New Capabilities for Acelerating Machine Learning Experiment](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-now-comes-with-new-capabilities-for-accelerating-machine-learning-experimentation/)
- Answer 004 [Why are my contributions not showing up on my profile](https://help.github.com/en/articles/why-are-my-contributions-not-showing-up-on-my-profile)
- Answer 005 When in doubt lower the learning rate to help the model converge.
- Answer 006 Some algorithms are sensitive to initialization - weight initialization, centroid initialization, use random_state, or random seed to get reproducible results
- Answer 007 You will gain intro level production deployment experience to AWS SageMaker, get experience with a capstone project, get intermediate-advanced python coding experience, OOP, docstring.
- Answer 008 [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
- Answer 009 [open AI gym reinforcement learning. Agents hide and seek](https://youtu.be/kopoLzvh5jY)
- Answer 010 There is a recommended format and page number should be substantial. [Capstone Report Template](https://github.com/udacity/machine-learning/blob/master/projects/capstone/capstone_report_template.md)
- Answer 011 [machine learning textbook online](http://aima.cs.berkeley.edu/)
- Answer 012 Checking the syllabus of each nanodegree is the best way to understand the detailed differences among those great choices.
- Answer 013 Check class imbalance, check curse of dimensionality (how many rows of data do you have), feature engineering, column selection, data cleaning, scaling, transformation, converting colored image to black and white? Best metrics. Benchmark is important!
- Answer 014 It is located in the Resource tab on the left. The resource tab is right underneath the search tab. [Read more about it here](http://bit.ly/aws-credit-udacity) Troubleshoot use this link.
- Answer 015 Practical benchmark for study projects like the capstone requires only a baseline model, such as a vanilla model, without hyperparameter tuning, an out-of-box model. The goal is to improve on this model and achieve a meaningful outcome at the end of the paper. Some students choose to apply a state-of-art new model on an existing paper and dataset. Some students choose to AlexNet as base, VGG as improvement or VGG as benchmark/ base and ResNet as improvement Improving a model from out-of-box default to hyperparameter tuned, or train with epochs, on custom dataset usually improves performance.
- Answer 016 The problem is that the p2 xlarge machine is no longer under EC2 category but now under sagemaker
	- visit https://console.aws.amazon.com/
	- top right corner click on support
	- click create a case (orange button)
	- Select Service Limit radio button
	- In the case, Search and Select SageMaker as Limit Type
	- Select the same region as the region that is displayed on the top right  corner of your amazon console.
	- Upon selected region, Select SageMaker Training as Resource Type
	- Select ml.p2.xlarge in Limit
	- New Limit Values `1`
Note GPU access is not available for all regions. Some students experienced errors.
- Answer 017 This ResourceLimitExceeded error can happen to `ml.m4.xlarge` as well as `ml.p2.xlarge` Full error message `ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateEndpoint operation: The account-level service limit 'ml.m4.xlarge for endpoint usage' is 0 Instances, with current utilization of 0 Instances and a request delta of 1 Instances. Please contact AWS support to request an increase for this limit.` Note specifically this is an `endpoint` resource error. Turns out there isn't good documentation about the actual limit of compute instances. And there is a difference between `sagemaker training` and `sagemaker hosting`. Definitely be sure to follow Answer 16 to request `sagemaker training` increase for `ml.p2.xlarge`, which is accelerated training. Be sure to request `sagemaker hosting` limit increase for `ml.m4.xlarge` if you encountered the above error. [Follow this link to resolve this issue](https://knowledge.udacity.com/questions/60402)
- Answer 018 [AWS Sagemaker instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/)
- Answer 019 More about the AWS Sagemaker compute instance limit issue. `Amazon SageMaker's limits for new accounts may differ from those listed under https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html#limits_sagemaker . Please contact customer service to request a limit increase for the resources you intend to use.
https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html#limits_sagemaker`
- Answer 020 Also make sure to conserve your resources by stopping Notebook instances asap, stop or delete endpoints on the [AWS console](https://console.aws.amazon.com/). 
- Answer 021 here's a great article on medium about how to get started with the capstone project. It also contain important tips [Tips for Udacity Machine Learning Nanodegree Capstone Project](https://medium.com/@mmatterr/tips-for-udacity-machine-learning-engineering-nanodegree-capstone-project-1110ce5b8cd0)
- Answer 022 AI and ML experience in Python is helpful. There's a free Python course offered by Udacity. Student also found the AI with Python nanodegree to be helpful. A review of Python and Jupyter Notebook is helpful. Preview capstone and think about datasets and examples to use from Kaggle way before the capstone is also said to be helpful.
- ANswer 023 The best way to compare is to download the syllabus PDF of each nanodegree and compare. On a high level: Intro to Machine Learning nanodegree covers the classical machine learning algorithms like decision trees, support vector machine, has a brief intro project computer vision neural network Convolutional Neural Network (CNN) and transfer learning, then introduces several unsupervised learning methods. Both project 1 and project 2 place heavy emphasis on data cleaning, feature engineering using pandas, scikit learn, and project 2 with PCA. MLEND focuses on Amazon cloud Amazon Web Service (AWS) SageMaker deployment, starting a notebook instance on AWS SageMaker, deploy a sentiment analysis model, try out some example notebooks, then there's a bit of feature engineering for natural language processing (NLP) engineering longest common sequence (LCS) and containtment, introduction to case studies in AWS SageMaker including deploying custom Pytorch models. Finally it ends with a capstone proposal and a capstone project, which can take quite a bit of time. The MLEND requires students to implement machine learning right away. Those new to ML should start with intro to ML. 
- ANswer 024 When in doubt try a smaller learning rate such as 0.01, 0.005, 0.001
- Answer 025 First suggestion is to check the model container and instance, test it out on a known dataset and see if it gives good result. This way we guarantee the model instance is pristine. Though this error you see is largely because even though XGBoost model trained (though it looks like it just have one tree one split), but the data csv or the data label contains error. For example, in this case, the target column was compromised during the cleaning process so it is either all 0 or all 1. The model didn't have to train per se, it can just predict always 0 or always 1 and achieve zero training error and zero validation error. Because the error is zero. it couldn't train on zero error.

# Related FAQ
- Project Amazon SageMaker Deployment 001
- SageMaker 001 003
- FAQ_ML 007 Related to FAQ Q 016
