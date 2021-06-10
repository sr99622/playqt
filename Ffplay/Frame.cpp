#include "Frame.h"

Frame::Frame() : QObject()
{

}

Frame::~Frame()
{
    if (pBGR != nullptr) {
        cudaFree(pBGR);
        cudaFree(pYUV[0]);
        cudaFree(pYUV[1]);
        cudaFree(pYUV[2]);
    }
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

Mat Frame::hwToMat()
{
    //Npp8u *pYUV[3], *pBGR;
    int ff = width * height;
    int hs = ff / 4;
    int ch = 3;

    Mat bgr(height, width, CV_8UC4);

    try {
        if (pBGR == nullptr) {
            eh.ck(cudaMalloc((void**)(&pYUV[0]), sizeof(Npp8u) * ff));
            eh.ck(cudaMalloc((void**)(&pYUV[1]), sizeof(Npp8u) * hs));
            eh.ck(cudaMalloc((void**)(&pYUV[2]), sizeof(Npp8u) * hs));
            eh.ck(cudaMalloc((void**)(&pBGR), sizeof(Npp8u) * ch * ff));
        }

        eh.ck(cudaMemcpy(pYUV[0], frame->data[0], sizeof(Npp8u) * ff, cudaMemcpyHostToDevice));
        eh.ck(cudaMemcpy(pYUV[1], frame->data[1], sizeof(Npp8u) * hs, cudaMemcpyHostToDevice));
        eh.ck(cudaMemcpy(pYUV[2], frame->data[2], sizeof(Npp8u) * hs, cudaMemcpyHostToDevice));

        //eh.ck(nppiYUV420ToBGR_8u_P3C4R(pYUV, frame->linesize, pBGR, ch * frame->linesize[0], {width, height}), "convert forwards");
        eh.ck(nppiYUV420ToRGB_8u_P3C3R(pYUV, frame->linesize, pBGR, ch * frame->linesize[0], {width, height}), "convert forwards");

        eh.ck(cudaMemcpy(bgr.data, pBGR, sizeof(Npp8u) * ch * ff, cudaMemcpyDeviceToHost));

        /*
        eh.ck(cudaFree(pYUV[0]));
        eh.ck(cudaFree(pYUV[1]));
        eh.ck(cudaFree(pYUV[2]));
        eh.ck(cudaFree(pBGR));
        */
    }
    catch (const exception &e) {
        cout << e.what() << endl;
    }

    return bgr;
}

void Frame::hwReadMat(const Mat& mat)
{
    //Npp8u *pYUV[3], *pBGR;
    int ff = width * height;
    int hs = ff / 4;
    int ch = 3;

    try {
        if (pBGR == nullptr) {
            eh.ck(cudaMalloc((void**)(&pYUV[0]), sizeof(Npp8u) * ff));
            eh.ck(cudaMalloc((void**)(&pYUV[1]), sizeof(Npp8u) * hs));
            eh.ck(cudaMalloc((void**)(&pYUV[2]), sizeof(Npp8u) * hs));
            eh.ck(cudaMalloc((void**)(&pBGR), sizeof(Npp8u) * ch * ff));
        }

        eh.ck(cudaMemcpy(pBGR, mat.data, sizeof(Npp8u) * ch * ff, cudaMemcpyHostToDevice));

        //eh.ck(nppiBGRToYUV420_8u_AC4P3R(pBGR, ch * frame->linesize[0], pYUV, frame->linesize, {width, height}), "convert backwards");
        eh.ck(nppiRGBToYUV420_8u_C3P3R(pBGR, ch * frame->linesize[0], pYUV, frame->linesize, {width, height}), "convert backwards");

        eh.ck(cudaMemcpy(frame->data[0], pYUV[0], ff, cudaMemcpyDeviceToHost), "cudaMemcpy");
        eh.ck(cudaMemcpy(frame->data[1], pYUV[1], hs, cudaMemcpyDeviceToHost), "cudaMemcpy");
        eh.ck(cudaMemcpy(frame->data[2], pYUV[2], hs, cudaMemcpyDeviceToHost), "cudaMemcpy");

        /*
        eh.ck(cudaFree(pYUV[0]));
        eh.ck(cudaFree(pYUV[1]));
        eh.ck(cudaFree(pYUV[2]));
        eh.ck(cudaFree(pBGR));
        */
    }
    catch (const exception &e) {
        cout << e.what() << endl;
    }
}

void Frame::readMat(const Mat& mat)
{
    width = mat.cols;
    height = mat.rows;
    int cvLinesizes[1];
    cvLinesizes[0] = mat.step1();
    if (frame == NULL) {
      frame = av_frame_alloc();
      av_image_alloc(frame->data, frame->linesize, width, height, AV_PIX_FMT_YUV420P, 1);
    }
    SwsContext *conversion = sws_getContext(width, height, AV_PIX_FMT_BGR24, width, height,
                                            (AVPixelFormat)frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, &mat.data, cvLinesizes, 0, height, frame->data, frame->linesize);
    sws_freeContext(conversion);
}

Mat Frame::toMat()
{
  Mat image(height, width, CV_8UC3);
  int cvLinesizes[1];
  cvLinesizes[0] = image.step1();
  SwsContext *conversion = sws_getContext(width, height, (AVPixelFormat)frame->format, width, height,
                                          AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
  sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
  sws_freeContext(conversion);
  return image;
}
