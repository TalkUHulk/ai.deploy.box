//
// Created by TalkUHulk on 2022/10/19.
//

#ifndef AIENGINE_ERRORCODE_H
#define AIENGINE_ERRORCODE_H

namespace AIDB {

    enum StatusCode {
        NO_ERROR           = 0,
        MODEL_CREATE_ERROR = 1,
        SESSION_CREATE_ERROR = 2,
        COMPUTE_SIZE_ERROR = 3,
        NO_EXECUTION       = 4,
        INVALID_VALUE      = 5,

        // User error
        INPUT_DATA_ERROR = 10,
        CALL_BACK_STOP   = 11,

        // Op Resize Error
        TENSOR_NOT_SUPPORT = 20,
        TENSOR_NEED_DIVIDE = 21,

        //
        NOT_IMPLEMENT = 100
    };
} // namespace MNN

#endif //AIENGINE_ERRORCODE_H
