//
// Created by TalkUHulk on 2023/7/3.
//

#ifndef AIDB_REFLECT_HPP
#define AIDB_REFLECT_HPP

#include <string>
#include <map>
#include <iostream>
#include "wrapper/AiDBBaseNode.hpp"

namespace AiDBServer{
    class NodeFactory {
    public:
        NodeFactory() = default;
        virtual ~NodeFactory() = default;
        virtual AiDBBaseNode* newInstance() = 0;
    };


    class Reflector {
    public:
        Reflector();
        ~Reflector();
        void registerNode(const std::string& className, NodeFactory *nf);
        AiDBBaseNode* getNewInstance(const std::string& className);
    private:
        std::map<std::string, NodeFactory*> _nodes_factory;
    };

    // reflector instances, global
    Reflector& reflector();


    #define AiDB_REGISTER(name)\
    class NodeFactory_##name : public NodeFactory{\
    public:\
        NodeFactory_##name() = default;\
        virtual ~NodeFactory_##name() = default;\
        AiDBBaseNode* newInstance() {\
            return new name(); \
        }\
    }; \
    class Register_##name{\
    public:\
        Register_##name(){\
            reflector().registerNode(#name, new NodeFactory_##name()); \
        }\
    };\
    Register_##name register_##name;


    template<typename T>
    T* getAiDBNode(const std::string& className) {
        return dynamic_cast<T*>(reflector().getNewInstance(className));
    }

}
#endif //AIDB_REFLECT_HPP
