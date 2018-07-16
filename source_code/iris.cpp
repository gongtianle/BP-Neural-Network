//
//  dataProc.cpp
//  BPNN
//
//  Created by Lnrd on 2018/6/14.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#include "iris.hpp"

void loadData(vector<sample>& samples, string filename){
    ifstream in(filename);
    sample temp;
    string line;
    while(!in.eof()){
        temp.in.clear();
        temp.out.clear();
        getline(in, line);
        if(line.compare("") == 0)
            continue;
        vector<string> v;
        split(v, line, ",");
        if(v[4].compare("Iris-setosa") == 0){
            temp.out.push_back(1.f);
            temp.out.push_back(0.f);
            temp.out.push_back(0.f);
        }
        else if(v[4].compare("Iris-versicolor") == 0){
            temp.out.push_back(0.f);
            temp.out.push_back(1.f);
            temp.out.push_back(0.f);
        }
        else{
            temp.out.push_back(0.f);
            temp.out.push_back(0.f);
            temp.out.push_back(1.f);
        }
        for(int i = 0; i < 4; i++)
            temp.in.push_back(atof(v[i].c_str()));
        if(temp.in.size() == 4 && temp.out.size() == 3)
            samples.push_back(temp);
    }
    in.close();
}
void max_min(const vector<sample>& samples, vector<double>& max, vector<double>& min){
    int n = int(samples.size());
    int m = int(samples[0].in.size());
    max.assign(m, -INFINITY);
    min.assign(m, INFINITY);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(samples[i].in[j] > max[j])
                max[j] = samples[i].in[j];
            if(samples[i].in[j] < min[j])
                min[j] = samples[i].in[j];
        }
    }
}
void normalize(vector<sample>& samples, const vector<double>& max, const vector<double>& min){
    int n = int(samples.size());
    int m = int(max.size());
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            samples[i].in[j] = (samples[i].in[j] - min[j]) / (max[j] - min[j]);
}
void normalize(sample& sample, const vector<double>& max, const vector<double>& min){
    int m = int(max.size());
    for(int i = 0; i < m; i++)
        sample.in[i] = (sample.in[i] - min[i]) / (max[i] - min[i]);
}
string translate_result(const vector<double>& v){
    int mi = max_index(v);
    string result;
    switch (mi) {
        case 0:
            result = "Iris-setosa";
            break;
        case 1:
            result = "Iris-versicolor";
            break;
        default:
            result = "Iris-virginica";
            break;
    }
    return result;
}
