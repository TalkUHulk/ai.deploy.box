//
// Created by TalkUHulk on 2023/6/12.
//

#ifndef AIDB_AIDBWRAPPER_HPP
#define AIDB_AIDBWRAPPER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include "AiDBFaceDetect.hpp"
#include "AiDBFaceLandMark.hpp"
#include "AiDBFace3DDFA.hpp"
#include "AiDBFaceParsing.hpp"
#include "AiDBYoloX.hpp"
#include "AiDBYoloV7.hpp"
#include "AiDBYoloV8.hpp"
#include "AiDBPPOcr.hpp"
#include "AiDBMoveNet.hpp"
#include "AiDBMobileVit.hpp"
#include "AiDBMobileStyleGan.hpp"
#include "AiDBAnimeGan.hpp"

namespace py = pybind11;


enum AIDB_ModelID{
    EMPTY = -1,
    SCRFD = 0,
    PFPLD = 1,
    TDDDFA = 2,
    BISENET = 3,
    MOVENET = 4,
    YOLOX = 5,
    YOLOV7 = 6,
    YOLOV8 = 7,
    PPOCR = 8,
    MOBILE_VIT = 9,
    MOBILE_STYLE_GAN = 10,
    ANIME_GAN = 11
};


class AiDBWrapper{

    void* _aidb{};
    AIDB_ModelID _model_id = EMPTY;
    void release(){
        switch (_model_id) {
            case SCRFD: {
                delete (AiDBFaceDetect*) _aidb;
                break;
            }
            case PFPLD: {
                delete (AiDBFaceLandMark*) _aidb;
                break;
            }

            case TDDDFA: {
                delete (AiDBFace3DDFA*) _aidb;
                break;
            };
            case BISENET: {
                delete (AiDBFaceParsing*) _aidb;
                break;
            };
            case MOVENET: {
                delete (AiDBMoveNet*) _aidb;
                break;
            };
            case YOLOX: {
                delete (AiDBYoloX*) _aidb;
                break;
            };
            case YOLOV7: {
                delete (AiDBYoloV7*) _aidb;
                break;
            };
            case YOLOV8: {
                delete (AiDBYoloV8*) _aidb;
                break;
            };
            case PPOCR: {
                delete (AiDBPPOcr*) _aidb;
                break;
            };
            case MOBILE_VIT: {
                delete (AiDBMobileVit*) _aidb;
                break;
            };
            case MOBILE_STYLE_GAN: {
                delete (AiDBMobileStyleGan*) _aidb;
                break;
            };
            case ANIME_GAN: {
                delete (AiDBAnimeGan*) _aidb;
                break;
            };
            default: {
                std::cout << "Not support.\n";
                break;
            }

        }
        _model_id = EMPTY;
    }
public:
    AiDBWrapper() = default;
    ~AiDBWrapper(){
        if(_aidb != nullptr) {
            release();
        }
        _aidb = nullptr;
    }
    int init(AIDB_ModelID model_id, const py::list& models, const py::list& backend, const char* config_path){

        if(EMPTY != model_id){
            release();
        }
        if(_aidb != nullptr){
            delete (AiDBMobileStyleGan*)_aidb;
            _aidb = nullptr;
        }
        _model_id = model_id;
        assert(models.size() == backend.size());

        switch (model_id) {
            case SCRFD:{
                _aidb = new AiDBFaceDetect(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                        config_path);
                break;
            }
            case PFPLD:{
                _aidb = new AiDBFaceLandMark(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                             models[1].cast<std::string>(), backend[1].cast<std::string>(),
                                                     config_path);
                break;
            }
                
            case TDDDFA:{
                _aidb = new AiDBFace3DDFA(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                          models[1].cast<std::string>(), backend[1].cast<std::string>(),
                                          config_path);
                break;
            };
            case BISENET:{
                _aidb = new AiDBFaceParsing(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                            models[1].cast<std::string>(), backend[1].cast<std::string>(),
                                            config_path);
                break;
            };
            case MOVENET:{
                _aidb = new AiDBMoveNet(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                       config_path);
                break;
            };
            case YOLOX:{
                _aidb = new AiDBYoloX(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                       config_path);
                break;
            };
            case YOLOV7:{
                _aidb = new AiDBYoloV7(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                       config_path);
                break;
            };
            case YOLOV8:{
                _aidb = new AiDBYoloV8(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                        config_path);
                break;
            };
            case PPOCR:{
                _aidb = new AiDBPPOcr(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                      models[1].cast<std::string>(), backend[1].cast<std::string>(),
                                      models[2].cast<std::string>(), backend[2].cast<std::string>(),
                                      config_path);
                break;
            };
            case MOBILE_VIT:{
                _aidb = new AiDBMobileVit(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                      config_path);
                break;
            };
            case MOBILE_STYLE_GAN:{
                _aidb = new AiDBMobileStyleGan(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                               models[1].cast<std::string>(), backend[1].cast<std::string>(),
                                               config_path);
                break;
            };
            case ANIME_GAN:{
                _aidb = new AiDBAnimeGan(models[0].cast<std::string>(), backend[0].cast<std::string>(),
                                               config_path);
                break;
            };
            default:{
                std::cout << "Not support.\n";
                break;
            }

        }

        if(nullptr == _aidb){
            return -1;
        }
        return 0;
    }
    py::dict forward(py::array_t<uint8_t>& frame_array, int frame_width, int frame_height){

        py::dict results_meta;

        switch (_model_id) {
            case SCRFD:{
                ((AiDBFaceDetect*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            }
            case PFPLD:{
                ((AiDBFaceLandMark*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            }

            case TDDDFA:{
                ((AiDBFace3DDFA*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case BISENET:{
                ((AiDBFaceParsing*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case MOVENET:{
                ((AiDBMoveNet*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case YOLOX:{
                ((AiDBYoloX*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case YOLOV7:{
                ((AiDBYoloV7*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case YOLOV8:{
                ((AiDBYoloV8*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case PPOCR:{
                ((AiDBPPOcr*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case MOBILE_VIT:{
                ((AiDBMobileVit*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            case ANIME_GAN:{
                ((AiDBAnimeGan*)_aidb)->forward(frame_array, frame_width, frame_height, results_meta);
                break;
            };
            default:{
                std::cout << "Not support.\n";
                break;
            }
        }

        return results_meta;

    }

    py::dict forward(py::array_t<float>& frame_array){

        py::dict results_meta;
        if(_model_id != MOBILE_STYLE_GAN){
            std::cout << "error.\n";
        }
        ((AiDBMobileStyleGan*)_aidb)->forward(frame_array, results_meta);
        return results_meta;

    }
};
#endif //AIDB_AIDBWRAPPER_HPP
