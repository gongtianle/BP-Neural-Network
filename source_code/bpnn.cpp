//
//  bpnn.cpp
//  BPNN
//
//  Created by Lnrd on 2018/6/13.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#include "bpnn.hpp"
inline double sigmoid(double x){
    return 1/(1 + exp(-x));
}
inline double genarateRandom(){
    return ((2.0*(double)rand()/RAND_MAX) - 1);
}

BPNN::BPNN(int in, int hn, int hl, int on, double alpha){
    INPUTNODE = in;
    HIDDENNODE = hn;
    HIDDENLAYER = hl;
    OUTPUTNODE = on;
    learningRate = alpha;
    inputLayer = new InputNode[INPUTNODE];
    outputLayer = new OutputNode[OUTPUTNODE];
    hiddenLayer = new HiddenNode*[HIDDENLAYER];
    for(int i=0; i < HIDDENLAYER; i++){
        hiddenLayer[i] = new HiddenNode[HIDDENNODE];
        for(int j = 0; j < HIDDENNODE; j++){
            hiddenLayer[i][j] = *new HiddenNode();
        }
    }
    for(int i = 0; i < INPUTNODE; i++)
    {
        inputLayer[i] = *new InputNode();
    }
    for(int i = 0; i < OUTPUTNODE; i++)
    {
        outputLayer[i] = *new OutputNode();
    }
    
}

BPNN::~BPNN(){
    delete[] inputLayer;
    delete[] outputLayer;
    for(int i=0; i < HIDDENLAYER; i++){
        delete[] hiddenLayer[i];
    }
    delete[] hiddenLayer;
}
void BPNN::initialize(){
    srand((unsigned)time(NULL));
    error = 100.f;
    //initialize input layer
    for(int i = 0; i < INPUTNODE; i++)
    {
        inputLayer[i].weight.clear();
        inputLayer[i].Delta_w.clear();
        for(int j = 0; j < HIDDENNODE; j++)
        {
            inputLayer[i].weight.push_back(genarateRandom());
            inputLayer[i].Delta_w.push_back(0.f);
        }
    }
    //initialize hidden layers
    for(int i = 0; i < HIDDENLAYER - 1; i++)
    {
        for(int j = 0; j < HIDDENNODE; j++){
            hiddenLayer[i][j].bias = genarateRandom();
            hiddenLayer[i][j].Delta_b = 0.f;
            hiddenLayer[i][j].weight.clear();
            hiddenLayer[i][j].Delta_w.clear();
            for(int k = 0; k < HIDDENNODE; k++){
                hiddenLayer[i][j].weight.push_back(genarateRandom());
                hiddenLayer[i][j].Delta_w.push_back(0.f);
            }
        }
    }
    for(int j = 0; j < HIDDENNODE; j++)
    {
        hiddenLayer[HIDDENLAYER - 1][j].bias = genarateRandom();
        hiddenLayer[HIDDENLAYER - 1][j].Delta_b = 0.f;
        hiddenLayer[HIDDENLAYER - 1][j].weight.clear();
        hiddenLayer[HIDDENLAYER - 1][j].Delta_w.clear();
        for (int k = 0; k < OUTPUTNODE; k++)
        {
            hiddenLayer[HIDDENLAYER - 1][j].weight.push_back(genarateRandom());
            hiddenLayer[HIDDENLAYER - 1][j].Delta_w.push_back(0.f);
        }
    }
    //initialize output layer
    for(int i = 0; i < OUTPUTNODE; i++)
    {
        outputLayer[i].bias = genarateRandom();
        outputLayer[i].Delta_b = 0.f;
    }
}
void BPNN::setInput(vector<double> sampleIn){
    for(int i = 0; i < INPUTNODE; i++)
    inputLayer[i].value = sampleIn[i];
}
void BPNN::setOutput(vector<double> sampleOut){
    for(int i = 0; i < OUTPUTNODE; i++)
    outputLayer[i].rightout = sampleOut[i];
}
void BPNN::forwardPropagation(){
    for(int j = 0; j < HIDDENNODE; j++)
    {
        double sum = 0.f;
        for(int k = 0; k < INPUTNODE; k++){
            sum += inputLayer[k].value * inputLayer[k].weight[j];
        }
        sum += hiddenLayer[0][j].bias;
        hiddenLayer[0][j].value = sigmoid(sum);
    }
    for(int i = 1; i < HIDDENLAYER; i++){
        for(int j = 0; j < HIDDENNODE; j++)
        {
            double sum = 0.f;
            for(int k = 0; k < HIDDENNODE; k++){
                sum += hiddenLayer[i - 1][k].value * hiddenLayer[i - 1][k].weight[j];
            }
            sum += hiddenLayer[i][j].bias;
            hiddenLayer[i][j].value = sigmoid(sum);
        }
    }
    for(int i = 0; i < OUTPUTNODE; i++){
        double sum = 0.f;
        for(int j = 0; j < HIDDENNODE; j++){
            sum += hiddenLayer[HIDDENLAYER - 1][j].value * hiddenLayer[HIDDENLAYER - 1][j].weight[i];
        }
        sum += outputLayer[i].bias;
        outputLayer[i].value = sigmoid(sum);
    }
}
void BPNN::backPropagation(){
    for(int i = 0; i < OUTPUTNODE; i++){
        error += outputLayer[i].rightout * log(outputLayer[i].value) + (1 - outputLayer[i].rightout) * log(1 - outputLayer[i].value);
    
        outputLayer[i].delta = (outputLayer[i].value - outputLayer[i].rightout)*(1-outputLayer[i].value)*outputLayer[i].value;
    }
    for(int j = 0; j < HIDDENNODE; j++){
        double sum = 0.f;
        for (int k = 0; k < OUTPUTNODE; k++){
            sum += outputLayer[k].delta * hiddenLayer[HIDDENLAYER - 1][j].weight[k];
        }
        hiddenLayer[HIDDENLAYER - 1][j].delta = sum * (1 - hiddenLayer[HIDDENLAYER - 1][j].value) * hiddenLayer[HIDDENLAYER - 1][j].value;
    }
    for(int i = HIDDENLAYER - 2; i >= 0; i--){
        for (int j = 0; j < HIDDENNODE; j++){
            double sum = 0.f;
            for (int k = 0; k < HIDDENNODE; k++){
                sum += hiddenLayer[i + 1][k].delta * hiddenLayer[i][j].weight[k];
            }
            hiddenLayer[i][j].delta = sum * (1 - hiddenLayer[i][j].value) * hiddenLayer[i][j].value;
        }
    }
    for(int i = 0; i < INPUTNODE; i++){
        for(int j = 0; j < HIDDENNODE; j++){
            inputLayer[i].Delta_w[j] += inputLayer[i].value * hiddenLayer[0][j].delta;
        }
    }
    for (int i = 0; i < HIDDENLAYER - 1; i++)
    {
        for (int j = 0; j < HIDDENNODE; j++){
            hiddenLayer[i][j].Delta_b += hiddenLayer[i][j].delta;
            for (int k = 0; k < HIDDENNODE; k++){
                hiddenLayer[i][j].Delta_w[k] += hiddenLayer[i][j].value * hiddenLayer[i + 1][k].delta;
            }
        }
    }
    for (int j = 0; j < HIDDENNODE; j++)
    {
        hiddenLayer[HIDDENLAYER - 1][j].Delta_b += hiddenLayer[HIDDENLAYER - 1][j].delta;
        for (int k = 0; k < OUTPUTNODE; k++){
            hiddenLayer[HIDDENLAYER - 1][j].Delta_w[k] += hiddenLayer[HIDDENLAYER - 1][j].value * outputLayer[k].delta;
        }
    }
    for(int i = 0; i < OUTPUTNODE; i++){
        outputLayer[i].Delta_b += outputLayer[i].delta;
    }
}
void BPNN::train(vector<sample> sampleGroup, double threshold){
    int sampleSize = int(sampleGroup.size());
	int count = 0;
    while(error > threshold){
        error = 0.f;
        for (int i = 0; i < INPUTNODE; i++)
        inputLayer[i].Delta_w.assign(inputLayer[i].Delta_w.size(), 0.f);
        for (int i = 0; i < HIDDENLAYER; i++){
            for (int j = 0; j < HIDDENNODE; j++){
                hiddenLayer[i][j].Delta_w.assign(hiddenLayer[i][j].Delta_w.size(), 0.f);
                hiddenLayer[i][j].Delta_b = 0.f;
            }
        }
        for (int i = 0; i < OUTPUTNODE; i++)
        outputLayer[i].Delta_b = 0.f;
        
        for (int i = 0; i < sampleSize; i++){
            setInput(sampleGroup[i].in);
            setOutput(sampleGroup[i].out);
            forwardPropagation();
            backPropagation();
            error = - error / sampleSize;
        }
        for(int i = 0; i < INPUTNODE; i++){
            for(int j = 0; j < HIDDENNODE; j++){
                    inputLayer[i].weight[j] -= learningRate * inputLayer[i].Delta_w[j] / sampleSize;
            }
        }
        for(int i = 0; i < HIDDENLAYER - 1; i++){
            for (int j = 0; j < HIDDENNODE; j++){
                hiddenLayer[i][j].bias -= learningRate * hiddenLayer[i][j].Delta_b / sampleSize;
                for (int k = 0; k < HIDDENNODE; k++){
                    hiddenLayer[i][j].weight[k] -= learningRate * hiddenLayer[i][j].Delta_w[k] / sampleSize;
                }
            }
        }
        for(int j = 0; j < HIDDENNODE; j++){
            hiddenLayer[HIDDENLAYER - 1][j].bias -= learningRate * hiddenLayer[HIDDENLAYER - 1][j].Delta_b / sampleSize;
            for(int k = 0; k < OUTPUTNODE; k++){
                hiddenLayer[HIDDENLAYER - 1][j].weight[k] -= learningRate * hiddenLayer[HIDDENLAYER - 1][j].Delta_w[k] / sampleSize;
            }
        }
        
        for (int i = 0; i < OUTPUTNODE; i++){
            outputLayer[i].bias -= learningRate * outputLayer[i].Delta_b / sampleSize;
        }
		count++;
		if (count > MAX_ITER) {
			break;
		}
    }
}
void BPNN::predict(vector<sample>&  testGroup){
    int testSize = int(testGroup.size());
    for(int iter = 0; iter < testSize; iter++)
    {
        testGroup[iter].out.clear();
        setInput(testGroup[iter].in);
        forwardPropagation();
        for (int i = 0; i < OUTPUTNODE; i++)
        testGroup[iter].out.push_back(outputLayer[i].value);
    }
}
void BPNN::predict(sample& sample){
    sample.out.clear();
    setInput(sample.in);
    forwardPropagation();
    for (int i = 0; i < OUTPUTNODE; i++)
    sample.out.push_back(outputLayer[i].value);
}
