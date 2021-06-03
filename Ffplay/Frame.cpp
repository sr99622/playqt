#include "Frame.h"

Frame::Frame() : QObject()
{

}

bool Frame::writable()
{
    if (!av_frame_is_writable(frame)) {
        if (av_frame_make_writable(frame) < 0) {
            cout << "unable to make frame writable" << endl;
            return false;
        }
    }
    return true;
}

void Frame::grayscale()
{
    if (!writable())
        return;

    int size = frame->width * frame->height;
    memset(frame->data[1], 128, size>>2);
    memset(frame->data[2], 128, size>>2);
}

void Frame::paintItBlack()
{
    if (!writable())
        return;

    int size = frame->width * frame->height;
    memset(frame->data[0], 0, size);
    memset(frame->data[1], 128, size>>2);
    memset(frame->data[2], 128, size>>2);
}

void Frame::fillPixel(int x, int y, const YUVColor &color)
{
    if (!writable())
        return;

    frame->data[0][y * frame->linesize[0] + x] = color.y;
    frame->data[1][(y>>1) * frame->linesize[1] + (x>>1)] = color.u;
    frame->data[2][(y>>1) * frame->linesize[2] + (x>>1)] = color.v;
}

void Frame::drawBox(const QRect &rect, int line_width, const YUVColor &color)
{
    //cout << "rect x: " << rect.x() << " y: " << rect.y() << " width: " << rect.width() << " height: " << rect.height() << endl;
    QMargins margins(1, 1, 1, 1);

    for (int i = 0; i < line_width; i++) {
        QRect border = rect - margins * i;
        int l = border.left();
        int r = border.right();
        int t = border.top();
        int b = border.bottom();

        l = min(l, width-1);
        l = max(l, 0);
        r = min(r, width-1);
        r = max(r, 0);
        t = min(t, height-1);
        t = max(t, 0);
        b = min(b, height-1);
        b = max(b, 0);

        for (int y = t; y < b; y++) {
            fillPixel(l, y, color);
            fillPixel(r, y, color);
        }
        for (int x = l; x < r; x++) {
            fillPixel(x, t, color);
            fillPixel(x, b, color);
        }
        fillPixel(r, b, color);
    }
}
