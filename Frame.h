#pragma once

#include <iostream>

extern "C" {
#include "libavformat/avformat.h"
}

#include <QObject>
#include <QRect>
#include <QMargins>

#include "yuvcolor.h"

using namespace std;

class Frame : public QObject
{
    Q_OBJECT

public:
    Frame();
    void paintItBlack();
    void grayscale();
    bool writable();
    void fillPixel(int x, int y, const YUVColor &color);
    void drawBox(const QRect &rect, int line_width, const YUVColor &color);

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

};

