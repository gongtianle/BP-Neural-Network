//
//  main.cpp
//  BPNN
//
//  Created by Lnrd on 2018/6/13.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#include <iostream>
#include <vector>
#include "bpnn.hpp"
#include "iris.hpp"
#include "cross_validation.hpp"
using namespace std;
int main(int argc, const char * argv[]) {
    vector<sample> v;
    loadData(v);
    int number_hidden_layers;
    int number_hidden_nodes;
    double learning_rate;
    cout << "Please input the number of hidden layers, " << endl << "and the number of nodes for each hidden layer:" << endl;
    cin >> number_hidden_layers >> number_hidden_nodes;
    cout << "Please input the learning rate:" << endl;
    cin >> learning_rate;
    BPNN net(4, number_hidden_nodes, number_hidden_layers, 3, learning_rate);
    cross_validate(net, v);
	system("pause");
    return 0;
}
