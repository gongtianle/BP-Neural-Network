//
//  util.cpp
//  BPNN
//
//  Created by Lnrd on 2018/6/14.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#include "util.hpp"
void split(vector<string>& v, const string& s, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}
