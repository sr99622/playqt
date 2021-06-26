#include "Frame.h"

Frame::Frame() : QObject()
{

}

Frame::Frame(int width, int height, const AVPixelFormat &pix_fmt)
{
    allocateFrame(width, height, pix_fmt);
}

Frame::~Frame()
{
    av_frame_free(&frame);
    if (pBGR != nullptr) {
        cudaFree(pBGR);
        cudaFree(pYUV[0]);
        cudaFree(pYUV[1]);
        cudaFree(pYUV[2]);
    }
}

void Frame::copy(Frame *vp)
{
    allocateFrame(vp->width, vp->height, (AVPixelFormat)vp->format);
    av_frame_copy(frame, vp->frame);
    serial = vp->serial;
    pts = vp->pts;
    duration = vp->duration;
    pos = vp->pos;
    width = vp->width;
    height = vp->height;
    format = vp->format;
    sar = vp->sar;
    uploaded = 0;
    flip_v = vp->flip_v;
}

void Frame::allocateFrame(int width, int height, const AVPixelFormat& pix_fmt)
{
    if (this->width == width && this->height == height && this->format == pix_fmt)
        return;

    if (frame != nullptr)
        av_frame_free(&frame);
    frame = av_frame_alloc();
    frame->width = width;
    frame->height = height;
    frame->format = pix_fmt;
    av_frame_get_buffer(frame, 32);
    av_frame_make_writable(frame);
    this->width = width;
    this->height = height;
    this->format = pix_fmt;
}

void Frame::pip(int ulc_x, int ulc_y, Frame *sub_vp)
{
    int pip_height = sub_vp->frame->height;
    int ulc = ulc_y * frame->linesize[0] + ulc_x;
    for (int y = 0; y < pip_height; y++)
        memcpy(frame->data[0] + ulc + y * frame->linesize[0], sub_vp->frame->data[0] + y * sub_vp->frame->linesize[0], sub_vp->frame->linesize[0]);

    int pip_u_height = sub_vp->frame->height >> 1;
    int u_ulc = (ulc_y >> 1) * frame->linesize[1] + (ulc_x >> 1);
    for (int y = 0; y < pip_u_height; y++) {
        memcpy(frame->data[1] + u_ulc + y * frame->linesize[1], sub_vp->frame->data[1] + y * sub_vp->frame->linesize[1], sub_vp->frame->linesize[1]);
        memcpy(frame->data[2] + u_ulc + y * frame->linesize[2], sub_vp->frame->data[2] + y * sub_vp->frame->linesize[2], sub_vp->frame->linesize[2]);
    }
}

void Frame::slice(int x, int y, Frame *sub_vp)
{
    int slice_height = sub_vp->frame->height;
    int offset = y * width + x;
    for (int i = 0; i < slice_height; i++)
        memcpy(sub_vp->frame->data[0] + (i * sub_vp->frame->linesize[0]), frame->data[0] + i * frame->linesize[0] + offset, sub_vp->frame->linesize[0]);

    int slice_u_height = slice_height >> 1;
    int u_offset = (y >> 1) * frame->linesize[1] + (x >> 1);
    for (int i = 0; i < slice_u_height; i++) {
        memcpy(sub_vp->frame->data[1] + (i * sub_vp->frame->linesize[1]), frame->data[1] + (i * frame->linesize[1]) + u_offset, sub_vp->frame->linesize[1]);
        memcpy(sub_vp->frame->data[2] + (i * sub_vp->frame->linesize[2]), frame->data[2] + (i * frame->linesize[2]) + u_offset, sub_vp->frame->linesize[2]);
    }

    sub_vp->frame->pts = pts;
    sub_vp->frame->pict_type = frame->pict_type;
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

        // RGB vs BGR such a waste of time
        //eh.ck(nppiYUV420ToBGR_8u_P3C4R(pYUV, frame->linesize, pBGR, ch * frame->linesize[0], {width, height}), "convert forwards");
        eh.ck(nppiYUV420ToRGB_8u_P3C3R(pYUV, frame->linesize, pBGR, ch * frame->linesize[0], {width, height}), "convert forwards");

        eh.ck(cudaMemcpy(bgr.data, pBGR, sizeof(Npp8u) * ch * ff, cudaMemcpyDeviceToHost));
    }
    catch (const exception &e) {
        cout << e.what() << endl;
    }

    return bgr;
}

void Frame::hwReadMat(const Mat& mat)
{
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

        // RGB vs BGR such a waste of time
        //eh.ck(nppiBGRToYUV420_8u_AC4P3R(pBGR, ch * frame->linesize[0], pYUV, frame->linesize, {width, height}), "convert backwards");
        eh.ck(nppiRGBToYUV420_8u_C3P3R(pBGR, ch * frame->linesize[0], pYUV, frame->linesize, {width, height}), "convert backwards");

        eh.ck(cudaMemcpy(frame->data[0], pYUV[0], ff, cudaMemcpyDeviceToHost), "cudaMemcpy");
        eh.ck(cudaMemcpy(frame->data[1], pYUV[1], hs, cudaMemcpyDeviceToHost), "cudaMemcpy");
        eh.ck(cudaMemcpy(frame->data[2], pYUV[2], hs, cudaMemcpyDeviceToHost), "cudaMemcpy");
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
                                          AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
  sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
  sws_freeContext(conversion);
  return image;
}
