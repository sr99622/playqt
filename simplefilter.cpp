#include "simplefilter.h"
#include "mainwindow.h"

SimpleFilter::SimpleFilter(QMainWindow *parent) : QObject(parent)
{
    mainWindow = parent;
    first_pass = true;
    rgb = NULL;
    img.data = NULL;
    sws_ctx = NULL;
}

SimpleFilter::~SimpleFilter()
{
    destroy();
}

void SimpleFilter::process(Frame *vp)
{
    if (!MW->mainPanel->controlPanel->engageFilter->isChecked()) {
        return;
    }

    infer(vp);
    //vp->grayscale();
    //processCPU(vp);
    //vp->paintItBlack();
    //QRect rect(200, 200, 200, 200);
    //YUVColor green(Qt::green);
    //vp->drawBox(rect, 10, green);
}

void SimpleFilter::processCPU(Frame *vp)
{
    cout << "frame pts: " << vp->pts << endl;

    if (img.data == NULL)
        img.data = (float*)malloc(sizeof(float) * vp->frame->width * vp->frame->height * 3);
    img.h = vp->height;
    img.w = vp->width;
    img.c = 3;
    for (int y = 0; y < vp->height; y++) {
        for (int x = 0; x < vp->frame->linesize[0]; x++) {
            int i = y * vp->frame->linesize[0] + x;
            img.data[i] = (float)vp->frame->data[0][i] / 255.0f;
        }
    }
    if (img.data != NULL) {
        free(img.data);
        img.data = NULL;
    }
}

void SimpleFilter::infer(Frame *vp)
{
    vector<bbox_t> result = MW->model->infer(vp, 0.2);
    for (int i = 0; i < result.size(); i++) {
        QRect rect(result[i].x, result[i].y, result[i].w, result[i].h);
        YUVColor green(Qt::green);
        vp->drawBox(rect, 1, green);
    }
}

void SimpleFilter::processGPU(Frame *vp)
{
}

void SimpleFilter::initialize(AVFrame *f)
{
    /*
    rgb = av_frame_alloc();
    rgb->width = f->width;
    rgb->height = f->height;
    rgb->format = AV_PIX_FMT_RGB24;
    av_frame_get_buffer(rgb, 32);
    av_frame_make_writable(rgb);

    sws_ctx = sws_getContext(f->width, f->height, (AVPixelFormat)f->format,
                             f->width, f->height, AV_PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);

    */
}

void SimpleFilter::destroy()
{
    if (img.data != NULL)
        free(img.data);
    //if (sws_ctx != NULL)
    //    sws_freeContext(sws_ctx);
    //if (rgb != NULL)
    //    av_frame_free(&rgb);
}

