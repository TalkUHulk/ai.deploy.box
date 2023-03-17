//
// Created by TalkUHulk on 2023/2/5.
//

#ifndef AIDEPLOYBOX_AIDBDATA_H
#define AIDEPLOYBOX_AIDBDATA_H

#include <vector>
#include <string>
#include <iostream>

namespace AIDB{
    typedef struct Bbox {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;

        float width() const{
            return x2 - x1;
        };
        float height() const{
            return y2 - y1;
        };
        float area() const{
            return width() * height();
        };
    }Bbox;


    struct FaceMeta: Bbox{
        std::vector<float> pose;
        std::vector<float> kps;
    };

    struct ObjectMeta: Bbox{
        int label;
    };

    template<typename T>
    struct PointT{
        PointT() = default;
        PointT(T x, T y):_x(x), _y(y){};

        PointT<T>& operator= (const PointT<T> &p){
            _x = p._x;
            _y = p._y;
            return *this;
        };

        T _x;
        T _y;
    };

    struct OcrMeta{
        std::vector<PointT<int>> box;
        std::string label;
        float conf;
        float conf_rotate;
    };

    struct ClsMeta{
        int label;
        std::string label_str;
        float conf;
    };

}


#endif //AIDEPLOYBOX_AIDBDATA_H
