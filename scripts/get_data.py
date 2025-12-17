"""
该模块用于获取微博情感分析数据集。
数据集包含100,000条样本，包含文本和情感标签。
数据集的标签分为正面（1）、负面（0）和中性（2）三类。
数据集的文本内容为中文微博评论，情感标签为整数类型。
数据集的下载链接为：https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k
数据集的格式为CSV文件，包含两列：'text'和'label'。
数据集的加载函数get_data()可以选择是否下载数据集，默认不下载。
加载数据集后，函数会打印数据集的前五行、形状、列名、描述性统计信息、相关信息以及缺失值占比统计。
如果下载数据集，则会将数据集保存到本地路径'../../data/weibo_senti_100k.csv'。
如果不下载数据集，则会从本地路径'../../data/weibo_senti_100k.csv'加载数据集。

usage:
python scripts/get_data.py
"""
import pandas as pd


def get_data(download: bool = False) -> pd.DataFrame:
    """
    Returns the Weibo sentiment analysis dataset.

    The dataset contains 100,000 samples with text and sentiment labels.

    Returns:
        pd.DataFrame: DataFrame containing the dataset with columns 'text' and 'label'.
    """
    if download:
        df = pd.read_csv("hf://datasets/dirtycomputer/weibo_senti_100k/weibo_senti_100k.csv")
        df.to_csv('../../data/raw/weibo_senti_100k.csv', index=False)
    else:
        df = pd.read_csv('../../data/raw/weibo_senti_100k.csv')
    print('数据加载完成')
    print('-----数据集前五行-----')
    print(df.head())
    print('-----数据集形状-----')
    print(df.shape)
    print('-----数据集列名-----')
    print(df.columns)
    print('-----数据集描述性统计-----')
    print(df.describe())
    print('-----数据集相关信息-----')
    print(df.info())
    print('-----缺失值占比统计-----')
    print((df.isnull().sum()/ df.shape[0]) * 100)
    return df


if __name__ == "__main__":
    data = get_data(False)

