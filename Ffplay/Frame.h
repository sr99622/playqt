#pragma once

#include <iostream>

extern "C" {
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
}
#include "opencv2/opencv.hpp"
#include <npp.h>

#include <QObject>
#include <QRect>
#include <QMargins>

#include "Utilities/yuvcolor.h"
#include "Utilities/cudaexception.h"

using namespace std;
using namespace cv;

class Frame : public QObject
{
    Q_OBJECT

public:
    Frame();
    ~Frame();
    void paintItBlack();
    void grayscale();
    bool writable();
    void fillPixel(int x, int y, const YUVColor &color);
    void drawBox(const QRect &rect, int line_width, const YUVColor &color);
    Mat toMat();
    void readMat(const Mat& mat);
    Mat hwToMat();
    void hwReadMat(const Mat& mat);

    Npp8u *pYUV[3] = {nullptr, nullptr, nullptr};
    Npp8u *pBGR = nullptr;

	AVFrame* frame;
	AVSubtitle sub;
	int serial;
	double pts;
	double duration;
	int64_t pos;
	int width;
	int height;
	int format;
	AVRational sar;
	int uploaded;
	int flip_v;

    CudaExceptionHandler eh;
};

