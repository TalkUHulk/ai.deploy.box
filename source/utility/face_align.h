//
// Created by Jack Yu on 23/03/2018.
//

#ifndef FACE_DEMO_FACEPREPROCESS_H
#define FACE_DEMO_FACEPREPROCESS_H

#include<opencv2/opencv.hpp>
#include "AIDBData.h"

namespace AIDB{

    // 112
    float arcface_src[5][2] = {
            { 30.2946f + 8.0f, 51.6963f },
            { 65.5318f + 8.0f, 51.5014f },
            { 48.0252f + 8.0f, 71.7366f },
            { 33.5493f + 8.0f, 92.3655f },
            { 62.7299f + 8.0f, 92.2041f }
    };
    // 512
    float ffhq_src[5][2] = {
            { 192.98138f, 239.94708f },
            { 318.90277f, 240.1936f },
            { 256.63416f, 314.01935f },
            { 201.26117f, 371.41043f },
            { 313.08905f, 371.15118f }
    };

    cv::Mat meanAxis0(const cv::Mat &src)
    {
        int num = src.rows;
        int dim = src.cols;

        // x1 y1
        // x2 y2

        cv::Mat output(1,dim,CV_32F);
        for(int i = 0 ; i <  dim; i ++)
        {
            float sum = 0 ;
            for(int j = 0 ; j < num ; j++)
            {
                sum+=src.at<float>(j,i);
            }
            output.at<float>(0,i) = sum/num;
        }

        return output;
    }

    cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
    {
        cv::Mat output(A.rows,A.cols,A.type());

        assert(B.cols == A.cols);
        if(B.cols == A.cols)
        {
            for(int i = 0 ; i <  A.rows; i ++)
            {
                for(int j = 0 ; j < B.cols; j++)
                {
                    output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
                }
            }
        }
        return output;
    }


    cv::Mat varAxis0(const cv::Mat &src)
    {
        cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
        cv::multiply(temp_ ,temp_ ,temp_ );
        return meanAxis0(temp_);

    }



    int MatrixRank(const cv::Mat& M)
    {
        cv::Mat w, u, vt;
        cv::SVD::compute(M, w, u, vt);
        cv::Mat1b nonZeroSingularValues = w > 0.0001;
        int rank = countNonZero(nonZeroSingularValues);
        return rank;

    }

//    References
//    ----------
//    .. [1] "Least-squares estimation of transformation parameters between two
//    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
//
//    """
//
//    Anthor:Jack Yu
    cv::Mat similarTransform(const cv::Mat& src,const cv::Mat& dst) {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat src_mean = meanAxis0(src);
        cv::Mat dst_mean = meanAxis0(dst);
        cv::Mat src_demean = elementwiseMinus(src, src_mean);
        cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        d.setTo(1.0f);
        if (cv::determinant(A) < 0) {
            d.at<float>(dim - 1, 0) = -1;

        }
        cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        cv::Mat U, S, V;
        cv::SVD::compute(A, S,U, V);

        // the SVD function in opencv differ from scipy .


        int rank = MatrixRank(A);
        if (rank == 0) {
            assert(rank == 0);

        } else if (rank == dim - 1) {
            if (cv::determinant(U) * cv::determinant(V) > 0) {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            } else {
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;

                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U* twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else{
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
            cv::Mat res = U* twp; // U
            T.rowRange(0, dim).colRange(0, dim) = U *  diag_ * V;
        }
        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d,S,res);
        float scale =  1.0/val*cv::sum(res).val[0];
        cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim) * src_mean.t();
        cv::Mat  temp2 = scale * temp1;
        cv::Mat  temp3 = dst_mean - temp2.t();
        T.at<float>(0,2) = temp3.at<float>(0);
        T.at<float>(1,2) = temp3.at<float>(1);
        T.rowRange(0, dim).colRange(0, dim) *= scale; // T[:dim, :dim] *= scale

        return T;
    }

    int faceAlign(const cv::Mat & img_src,
                  cv::Mat& face_aligned,
                  const std::shared_ptr<AIDB::FaceMeta>& face_meta,
                  int target_size = 112, const std::string& mode="ffhq") {
        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return 1;
        }
        if (face_meta->kps.empty()) {
            std::cout << "keypoints empty." << std::endl;
            return 2;
        }

        cv::Mat src_mat;
        cv::Mat dst_mat;
        if(face_meta->kps.size() == 10){
            src_mat = cv::Mat(5, 2, CV_32FC1, face_meta->kps.data());
        } else{
            float points[5][2] = {
                    {face_meta->kps[2 * 96], face_meta->kps[96 * 2 + 1]},
                    {face_meta->kps[2 * 97], face_meta->kps[2 * 97 + 1]},
                    {face_meta->kps[2 * 54], face_meta->kps[54 * 2 + 1] },
                    {face_meta->kps[2 * 88], face_meta->kps[88 * 2 + 1]},
                    {face_meta->kps[2 * 92], face_meta->kps[92 * 2 + 1]}
            };
            src_mat = cv::Mat(5, 2, CV_32FC1, points);
        }

        if("ffhq" == mode){
            dst_mat = cv::Mat(5, 2, CV_32FC1, ffhq_src) * target_size / 512;
        } else{
            dst_mat = cv::Mat(5, 2, CV_32FC1, arcface_src) * target_size / 112;
        }

        cv::Mat transform = similarTransform(src_mat, dst_mat);

        face_aligned.create(target_size, target_size, CV_32FC3);

        cv::Mat transfer_mat = transform(cv::Rect(0, 0, 3, 2));
        cv::warpAffine(img_src, face_aligned, transfer_mat,
                       cv::Size(target_size, target_size), cv::INTER_LINEAR, cv::BORDER_REFLECT);

        return 0;
    }

}
#endif //FACE_DEMO_FACEPREPROCESS_H
