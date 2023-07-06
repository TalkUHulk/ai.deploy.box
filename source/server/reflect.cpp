//
// Created by TalkUHulk on 2023/7/3.
//

#include "reflect.hpp"

namespace AiDBServer {
    Reflector &reflector() {
        static Reflector reflector;
        return reflector;
    }


    Reflector::Reflector() = default;


    Reflector::~Reflector() {
        for (auto &it : _nodes_factory) {
            delete it.second;
        }
        _nodes_factory.clear();
    }


    void Reflector::registerNode(const std::string &className, NodeFactory *nf) {
        auto it = _nodes_factory.find(className);
        if (it != _nodes_factory.end()) {
            //        std::cout << className << "该类已经存在……" << std::endl;
        } else {
            _nodes_factory[className] = nf;
        }
    }


    AiDBBaseNode *Reflector::getNewInstance(const std::string &className) {
        auto it = _nodes_factory.find(className);
        if (it != _nodes_factory.end()) {
            NodeFactory *nf = it->second;
            return nf->newInstance();
        }
        return nullptr;
    }
}