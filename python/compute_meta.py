# coding: utf-8
from read_data import read_data

def run_meta(chapters, sentences, char_list):
    """Compute various metadata about characters in char_list"""
    classifier_data_dict = read_data()
    # print(type(chapters))
