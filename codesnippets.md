## Environment Setup
* The easiest way to set up is install Anaconda on Windows or Mac (easy access to CUDA GPU included). [Anaconda Environment Cheatsheet](https://medium.com/data-science-bootcamp/anaconda-miniconda-cheatsheet-for-data-scientists-2c1be12f56db)

## Pandas
* pandas cheatsheet (Medium member premium or Browser Incognito Mode, if still have trouble reach out to mentor) [pandas cheatsheet](https://medium.com/@uniqtech/pandas-data-analysis-cheatsheet-ea619fd35b8f)
* important data structure: DataFrame and Series 
* pd.read_csv with separator delimiter comma `dataframe = pd.read_csv('data.csv', sep=",")` 
* access all the columns to loop over pd.DataFrame.columns
* one hot encoding pd.get_dummies()
* Use named variable to make your code more readable example `n` `df.head(n=2)` view pandas dataframe head with `n=2`, first two rows only
* Turn a list into a Series or a column my_series = pd.Series([1,2,3,4])
* Set a column of a dataframe equal to a series my_df['column_name'] = my_series
* Get row col count of dataframe df.shape
* Get summary data df.describe()

## Udacity Project Workspace
* View file view menu in Jupyter Notebook and Udacity workspace [youtube link](https://youtu.be/LhFJ8xPUNWg)

## Project walkthrough
* ? checking with mentor ops if okay to do end-to-end project walkthrough
* Review files in a working directly command line in Jupyter Notebook. Useful if want to view visuals module in CharityML and Customer Segmentation via data summary files markdown files. [youtube video](https://youtu.be/LhFJ8xPUNWg)
* Convert between numpy and pytorch tensor [youtube here](https://youtu.be/-mbFczHtOYA)

## Review
* 

## Numpy and Matplotlib and Data Visualization with Seaborn
* [Numpy and Matplotlib tutorial](http://cs231n.github.io/python-numpy-tutorial/)
* Key to display seaborn bar charts side-by-side in the same x axis and y axis using the dimension 'hue'
* [Intermediate plotting in matplotlib](http://cs231n.github.io/python-numpy-tutorial/) 

## Sklearn
* [sklearn cheatsheet (view w premium Medium membership or Browser incognito mode, contact mentor if issues)](https://medium.com/data-science-bootcamp/scikit-learn-sklearn-cheatsheet-72739349da70)

## Computer Vision and Image Classification
### custom argparse model arch parameter
### access the number of in_features in vgg16 and in_features in alexnet. They are 25088 and 9216
'''
if arg.arch == "vgg16":
    classifier_feature_num = model.classifier[0].in_features
    
elif arg.arch == "alexnet":
    classifier_feature_num = model.classifier[1].in_features
'''
- provide a default value for argparse parameters `parser.add_argument('--epochs', type=int. default=3, help='Epochs for training as int')`` [documentation](https://docs.python.org/3/library/argparse.html#default)
- Set the `in_feature` and `out_feature` of the classifier in transfer learning [Medium article explains transfer learning](https://medium.com/data-science-bootcamp/transfer-learning-with-pytorch-code-snippet-load-a-pretrained-model-900374950004)
```
def get_features(model):
    '''
    Get number of features for both model types
    '''
    if model.name == 'vgg16':
        return model.classifier[0].in_features
    else:
        return model.classifier[1].in_features
```
- Add an argument to argparse
```
parser.add_argument('--model', 
                    type=str, 
                    help='Choose vgg16 or alexnet from torchvision.models')
```