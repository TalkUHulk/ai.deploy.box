//
// Created by TalkUHulk on 2023/6/13.
//

#ifndef AIDB_AIDBMOBILESTYLEGAN_HPP
#define AIDB_AIDBMOBILESTYLEGAN_HPP

class AiDBMobileStyleGan{
    AIDB::Interpreter *mapping_ins{};
    AIDB::Interpreter *syn_ins{};
public:
    AiDBMobileStyleGan() = default;
    ~AiDBMobileStyleGan() {
        if(mapping_ins != nullptr){
            AIDB::Interpreter::releaseInstance(mapping_ins);
            mapping_ins = nullptr;
        }
        if(syn_ins != nullptr){
            AIDB::Interpreter::releaseInstance(syn_ins);
            syn_ins = nullptr;
        }
    }
    AiDBMobileStyleGan(const std::string& model1, const std::string& backend1,
                       const std::string& model2, const std::string& backend2,
                       const std::string& config_zoo){
        mapping_ins = AIDB::Interpreter::createInstance(model1, backend1, config_zoo);
        syn_ins = AIDB::Interpreter::createInstance(model2, backend2, config_zoo);
    }

    void forward(py::array_t<float>& frame_array, py::dict &result) {
        py::buffer_info buf = frame_array.request();
        auto* latent_ptr = (float*)buf.ptr;

        std::vector<std::vector<float>> outputs_map;
        std::vector<std::vector<int>> outputs_shape_map;

        mapping_ins->forward(latent_ptr, 512, 1, 1, outputs_map, outputs_shape_map);

        std::vector<std::vector<float>> outputs_syn;
        std::vector<std::vector<int>> outputs_shape_syn;

        syn_ins->forward((float*)outputs_map[0].data(), 512, 1, 1, outputs_syn, outputs_shape_syn);

        cv::Mat generated;
        AIDB::Utility::stylegan_post_process(generated, outputs_syn[0], outputs_shape_syn[0]);

        result["G"] = py::array_t<uint8_t>({ generated.rows,generated.cols, 3 }, generated.data);
        result["code"] = 0;
        result["msg"] = "succeed";
    }

};

#endif //AIDB_AIDBMOBILESTYLEGAN_HPP
