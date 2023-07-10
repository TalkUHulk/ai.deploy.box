package main

/*
#cgo CFLAGS : -I./include
#cgo LDFLAGS: -L./lib -lAiDB_C -Wl,-rpath,lib
#include <stdio.h>
#include <stdlib.h>
#include "aidb_c_api.h"
*/
import "C"
import (
	"encoding/json"
	"encoding/base64"
	"fmt"
	"unsafe"
	"os"
	"io/ioutil"
)

type GoAiDB struct {
	aidb C.AiDB
}

type AIDBInput struct {
	FlowUUID string   `json:"flow_uuid"`
	Backend  []string `json:"backend"`
	Model    []string `json:"model"`
	Zoo      string   `json:"zoo"`
}

type AIDBFaceOutput struct {
	BBox     []float32    `json:"bbox"`
	Conf     float32      `json:"conf"`
	LandMark [][2]float32 `json:"landmark"`
	Parsing  string       `json:"parsing"`
}

type AIDBObjectOutput struct {
	BBox  []float32 `json:"bbox"`
	Conf  float32   `json:"conf"`
	Label int       `json:"label"`
}

type AIDBOCROutput struct {
	Box        [][2]float32 `json:"box"`
	Conf       float32      `json:"conf"`
	ConfRotate float32      `json:"conf_rotate"`
	Label      string       `json:"label"`
}

type AIDBClsOutput struct {
	Conf  float32 `json:"conf"`
	Label int     `json:"label"`
}

type AiDBOutput struct {
	Code      int                `json:error_code`
	Face      []AIDBFaceOutput   `json:"face"`
	Object    []AIDBObjectOutput `json:"object"`
	Ocr       []AIDBOCROutput    `json:"ocr"`
	Cls       []AIDBClsOutput    `json:"cls"`
	Anime     string             `json:"anime"`
	Tddfa     string             `json:"tddfa"`
	KeyPoints [][2]float32       `json:"key_points"`
}

func readBinaryFile(filename string) ([]byte, error) {
    data, err := ioutil.ReadFile(filename)
    if err != nil {
        return nil, err
    }

    return data, nil
}

func AiDBRegister(A GoAiDB, flow_uuid string, model []string, backend []string, zoo string) int {

	aidb_input := AIDBInput{
		FlowUUID: flow_uuid,
		Backend:  backend,
		Model:    model,
		Zoo:      zoo,
	}

	aidb_input_str, err := json.Marshal(&aidb_input)

	if err != nil {
		fmt.Printf("序列号错误 err = %v\n", err)
	}
	fmt.Printf("序列化后= %v\n", string(aidb_input_str))

    c_aidb_input_str := C.CString(string(aidb_input_str))
    ret := C.AiDBRegister(A.aidb, c_aidb_input_str)
    C.free(unsafe.Pointer(c_aidb_input_str))

// 	ret := C.AiDBRegister(A.aidb, (*C.char)(unsafe.Pointer(&aidb_input_str[0])))

	return int(ret)
}

func AiDBUnRegister(A GoAiDB, flow_uuid string) int {

	c_flow_uuid := C.CString(flow_uuid)
	ret := C.AiDBUnRegister(A.aidb, c_flow_uuid)
	defer C.free(unsafe.Pointer(c_flow_uuid))
	return int(ret)
}

func AiDBCreate() GoAiDB {
	var ret GoAiDB
	ret.aidb = C.AiDBCreate()
	return ret
}

func AiDBFree(A GoAiDB) {
	C.AiDBFree(A.aidb)
}

func AiDBForward(A GoAiDB, flow_uuid string, image_path string) {
	c_flow_uuid := C.CString(flow_uuid)
    c_image_path := C.CString(image_path)
    c_size_out := (C.int)(0)
    size_in := 1024 * 1024

    c_size_in := (C.int)(size_in)
    c_result := (*C.char)(C.malloc((C.ulong)(size_in)))

    aidb_output := AiDBOutput{}

    ret := C.AiDBForward(A.aidb, c_flow_uuid, c_image_path, c_result, c_size_in, &c_size_out)

    if ret != 0 {
        fmt.Println("forward failed:", ret)
    }
    err := json.Unmarshal([]byte(C.GoString(c_result)), &aidb_output)
    if err != nil {
        fmt.Println("反序列化失败", err)
        return
    }

    if len(aidb_output.Tddfa) != 0{
        fmt.Println("Tddfa")
        decodedBytes, err := base64.StdEncoding.DecodeString(aidb_output.Tddfa)
        if err != nil {
              fmt.Println("Error decoding string:", err)
              return
        }

        fo, err := os.Create("./tddfa.jpg")
        if err != nil {
            fmt.Println("文件创建失败", err.Error())
            return
        }

        defer func() {
                if err := fo.Close(); err != nil {
                    panic(err)
                }
            }()

       if _, err := fo.Write(decodedBytes); err != nil {
                   panic(err)
               }
    }

    defer C.free(unsafe.Pointer(c_flow_uuid))
    defer C.free(unsafe.Pointer(c_result))
    defer C.free(unsafe.Pointer(c_image_path))
    fmt.Println(c_size_out)
}

func main() {
    imageRaw, err := readBinaryFile("./beckham.jpg")
    if err != nil {
        fmt.Println("read failed")
        return
    }
    imageBase64 := base64.StdEncoding.EncodeToString(imageRaw)

	ins := AiDBCreate()
	flow_uuid := "3ddfa-test"
	model := []string{"scrfd_500m_kps", "3ddfa_mb05_bfm_dense"}
	backend := []string{"mnn", "mnn"}
	zoo := "./config"
	ret := AiDBRegister(ins, flow_uuid, model, backend, zoo)
	if ret != 0 {
    	fmt.Println("register failed:", ret)
    	return
    }
    AiDBForward(ins, flow_uuid, imageBase64)
	AiDBFree(ins)
}

// source /etc/profile
// source ~/.profile