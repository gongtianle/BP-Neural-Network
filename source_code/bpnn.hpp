//
//  bpnn.hpp
//  BPNN
//
//  Created by Lnrd on 2018/6/13.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#ifndef bpnn_hpp
#define bpnn_hpp

#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>

using namespace std;
#define MAX_ITER 100000
/*functions*/
inline double sigmoid(double);
inline double genarateRandom();//generates a random number in [-1,1]


/*structs & classes*/
struct InputNode{
    double value;
    vector<double> weight;
    vector<double> Delta_w; //accumulated delta of weight
};
struct OutputNode{
    double value;
    double delta;
    double rightout;
    double bias;
    double Delta_b; //accumulated delta of bias
};
struct HiddenNode{
    double value;
    double delta;
    double bias;
    double Delta_b;
    vector<double> weight;
    vector<double> Delta_w;
};
struct sample{
    vector<double> in;
    vector<double> out;
};

class BPNN
{
public:
    BPNN(int=2,int=4,int=1,int=1,double=0.9);
    ~BPNN();
    void initialize();
    void forwardPropagation();
    void backPropagation();
    void train(vector<sample>, double=0.001);
    void predict(vector<sample>&);
    void predict(sample&);
    void setInput(vector<double>);
    void setOutput(vector<double>);
    
public:
    double error;
    int INPUTNODE;
    int HIDDENNODE;
    int HIDDENLAYER;
    int OUTPUTNODE;
    double learningRate;
    InputNode* inputLayer;
    OutputNode* outputLayer;
    HiddenNode** hiddenLayer;
};
#endif /* bpnn_hpp */
