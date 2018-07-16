//
//  dataProc.hpp
//  BPNN
//
//  Created by Lnrd on 2018/6/14.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#ifndef dataProc_hpp
#define dataProc_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "bpnn.hpp"
#include "util.hpp"

#define DATADIRECT "iris.txt"
using namespace std;
void loadData(vector<sample>&, string=DATADIRECT);
void max_min(const vector<sample>&, vector<double>&, vector<double>&);
void normalize(vector<sample>&, const vector<double>&, const vector<double>&);
void normalize(sample&, const vector<double>&, const vector<double>&);
string translate_result(const vector<double>&);
#endif /* dataProc_hpp */
