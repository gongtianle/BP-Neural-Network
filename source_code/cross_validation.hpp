//
//  cross_validation.hpp
//  BPNN
//
//  Created by Lnrd on 2018/7/14.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#ifndef cross_validation_hpp
#define cross_validation_hpp

#include <stdio.h>
#include "bpnn.hpp"
#include "iris.hpp"
#include<iomanip>
#include <algorithm>
# define THRESHOLD 0.001
void cross_validate(BPNN&, vector<sample>&);
#endif /* cross_validation_hpp */
