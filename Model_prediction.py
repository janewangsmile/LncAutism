#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:42:19 2019

@author: jun
"""
import os
from Bio import SeqIO
import pandas as pd
import regex as re
import numpy as np
from sklearn import preprocessing
import pickle

import argparse


def Model_prediction(inputFile, outputFile):
    IDs = []
    Seqs = []
    # 
    for record in SeqIO.parse("lncRNAs.fa","fasta"):
        ID = record.id
        Seq = str(record.seq)
        Seq = Seq.upper()
        
        IDs.append(ID)
        Seqs.append(Seq)
    #
    IDs = pd.DataFrame(IDs)
    IDs.columns = ['ENSG_ID']
    Seqs = pd.DataFrame(Seqs)
    Seqs.columns=['Sequence']
    lncRNAs = pd.concat([IDs,Seqs],axis=1)
    
    # filter the ones not in BrainSpan:
    lncRNAs_encoded_exp = pd.read_csv("lncRNAs_encoded_expression_features.csv",header=0)
    lncRNAs = lncRNAs.loc[lncRNAs['ENSG_ID'].isin(lncRNAs_encoded_exp['seq_name'])]
    
    # exp:
    exp = lncRNAs_encoded_exp.loc[lncRNAs_encoded_exp['seq_name'].isin(lncRNAs['ENSG_ID'])]
    lncRNAs.set_index('ENSG_ID',inplace=True)
    exp.set_index('seq_name',inplace = True)
    lncRNAs = pd.concat([lncRNAs,exp],axis=1,sort=False)
        
    # kmers:
    All_kmers = pd.read_csv("RF_kmer_feature_importance.csv",header=0)
    Top_kmers = list(All_kmers['0'][0:25])
    
    kmers = []
    for line in lncRNAs['Sequence']:
        feature = [len(re.findall(x,line,overlapped=True))/len(line) for x in Top_kmers]
        kmers.append(feature)
    
    kmers = pd.DataFrame(kmers)
    kmers.columns=Top_kmers
    kmers.index = lncRNAs.index
    
    # normalization:
    min_max_scaler = preprocessing.MinMaxScaler()
    kmers = min_max_scaler.fit_transform(kmers)
    kmers = pd.DataFrame(kmers)
    kmers.index = lncRNAs.index
    
    # combine:
    lncRNAs = pd.concat([lncRNAs,kmers],axis=1,sort=False)
    lncRNAs = lncRNAs.drop(['Sequence'],axis=1)
    #
    lncRNAs = min_max_scaler.fit_transform(lncRNAs)
    
    #prediction
    outFile = open(outputFile,'a')
    outFile.write("ENSG_ID\tLR\tSVM\tRF\n")
    # load models for prediction
    

    results=[]    
    models = ['LR','SVM','RF']
    for name in models:
        predictions = []
        for i in range(1,11,1):
            model = 'models/'+name+'_'+str(i)+'.sav'
            clf = pickle.load(open(model,'rb'))
            pred = clf.predict_proba(lncRNAs)
            predictions.append(pred[:,1])
        predictions = np.array(predictions)
        predictions = np.transpose(predictions)
        predictions = pd.DataFrame(predictions)
        predictions['mean'] = predictions.mean(axis=1)
        results.append(predictions['mean'])
    ### output results
    results = np.array(results)
    results = np.transpose(results)
    results = pd.DataFrame(results)
    results.index = kmers.index
    results.columns = models
    results.index.name = 'Gene_ID'
    
    results.to_csv(outputFile,index=True,header=True)
    
    
        
        
    

def main():
    parser = argparse.ArgumentParser(description="Program usage")
    parser.add_argument("-i","--fa",type=str, help="Input fasta file")
    parser.add_argument("-o","--csv",type=str,help="Prediction output")
    args=parser.parse_args()
    inputFile = args.fa
    outputFile = args.csv
    Model_prediction(inputFile,outputFile)
    

if __name__ == '__main__':
    main()

    
    
    
    
    
    



