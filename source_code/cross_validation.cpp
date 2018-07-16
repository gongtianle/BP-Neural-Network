//
//  cross_validation.cpp
//  BPNN
//
//  Created by Lnrd on 2018/7/14.
//  Copyright © 2018年 LNRD. All rights reserved.
//

#include "cross_validation.hpp"

void cross_validate(BPNN& net, vector<sample>& v){
    time_t start,stop;
    random_shuffle(v.begin(), v.end());
    vector<double> max;
    vector<double> min;
    max_min(v, max, min);
    int partition_size = (int) v.size() / 10;
    int i = 0;
    vector<sample> training_set;
    vector<sample> test_set;
    double average_prec = 0.f;
    cout << endl <<  "start cross validating..." << endl;
    cout << "------------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------" << endl;
    cout << "iteration\t|\t1\t|\t2\t|\t3\t|\t4\t|\t5\t|\t6\t|\t7\t|\t8\t|\t9\t|\t10\t" << endl;
    cout << "------------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------" << endl;
    cout << "precision\t" ;
    cout.flush();
    start = time(NULL);
    for(int iter = 0; iter < 10; iter++){
        training_set.clear();
        test_set.clear();
        for(i = 0; i < iter * partition_size; i++){
            training_set.push_back(v[i]);
        }
        for(i = iter * partition_size; i < (iter + 1) * partition_size; i++){
            test_set.push_back(v[i]);
        }
        for(i = (iter + 1) * partition_size; i < partition_size * 10; i++){
            training_set.push_back(v[i]);
        }
        net.initialize();
        normalize(training_set, max, min);
        net.train(training_set, THRESHOLD);
        double prec = 0.f;
        for(i = 0; i < test_set.size(); i++){
            string accurate = translate_result(test_set[i].out);
            normalize(test_set[i], max, min);
            net.predict(test_set[i]);
            string prediction = translate_result(test_set[i].out);
            if(accurate.compare(prediction) == 0){
                prec += 1.f;
            }
        }
        prec /= test_set.size();
        cout << "| " << setprecision(3) << prec * 100 << "%\t";
        cout.flush();
        average_prec += prec;
    }
    stop = time(NULL);
    cout << endl << "------------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------" << endl;
    average_prec /= 10;
    cout <<  "Takes " << (stop - start) << " seconds" << endl;
    cout <<  "Average precision: " << average_prec * 100 << "%" << endl << endl;
}
