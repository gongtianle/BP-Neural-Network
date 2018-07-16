//
//  util.hpp
//  BPNN
//
//  Created by Lnrd on 2018/6/14.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#ifndef util_hpp
#define util_hpp

#include <iostream>
#include <vector>
using namespace std;
void split(vector<string>&, const string&, const string&);
template<typename T>
int max_index(const vector<T>& v){
    int max_index = 0;
    for(int i = 1; i < v.size(); i++)
        if(v[i] > v[max_index])
            max_index = i;
    return max_index;
};
template<typename T>
int min_index(const vector<T>& v){
    int min_index = 0;
    for(int i = 1; i < v.size(); i++)
        if(v[i] < v[min_index])
            min_index = i;
    return min_index;
};
#endif /* util_hpp */
