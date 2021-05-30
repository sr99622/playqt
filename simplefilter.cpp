#include "simplefilter.h"
#include "mainwindow.h"
#include <ImagesCPU.h>
#include <ImageIO.h>

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

    //box_filter(vp);
    //cuda_example(vp);
    infer(vp);
    //processGPU(vp);
    //processCPU(vp);
}

void SimpleFilter::box_filter(Frame *vp)
{
    int width = vp->width;
    int height = vp->height;
    int frame_size = width * height;

    Npp8u *pSrc;

    if (!vp->writable())
        return;

    try {
        eh.ck(cudaMalloc((void**)(& pSrc), sizeof(Npp8u) * frame_size), "Allocate device source buffer");
        eh.ck(cudaMemcpy(pSrc, vp->frame->data[0], sizeof(Npp8u) * frame_size, cudaMemcpyHostToDevice), "Copy frame picture data buffer to device");
        eh.ck(nppsSet_8u(0, pSrc, frame_size), "set frame to 0");
        eh.ck(cudaMemcpy(vp->frame->data[0], pSrc, sizeof(Npp8u) * frame_size, cudaMemcpyDeviceToHost), "Copy result from device to frame");
        eh.ck(cudaFree(pSrc));
    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }
}

void SimpleFilter::processGPU(Frame *vp)
{
    int width = vp->width;
    int height = vp->height;
    int frame_size = width * height;

    Npp8u *pSrc;
    Npp32f *pNorm;
    image_t img;
    img.h = vp->frame->height;
    img.w = vp->frame->width;
    img.c = 3;
    img.data = (float*)malloc(sizeof(float) * img.w * img.h * 3);

    try {
        eh.ck(cudaMalloc((void**)(& pSrc), sizeof(Npp8u) * frame_size), "Allocate device source buffer");
        eh.ck(cudaMemcpy(pSrc, vp->frame->data[0], sizeof(Npp8u) * frame_size, cudaMemcpyHostToDevice), "Copy frame picture data buffer to device");
        eh.ck(cudaMalloc((void **)(&pNorm), sizeof(Npp32f) * frame_size), "Allocate a float buffer version of source for device");
        eh.ck(nppsConvert_8u32f(pSrc, pNorm, frame_size), "Convert frame date to float");
        eh.ck(nppsDivC_32f_I(255.0f, pNorm, frame_size), "normalize frame data");
        eh.ck(cudaMemcpy(img.data, pNorm, sizeof(Npp32f) * frame_size, cudaMemcpyDeviceToHost), "copy normalized frame data to img");
        eh.ck(cudaFree(pNorm));;
        eh.ck(cudaFree(pSrc));
    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }

    free(img.data);
    cout << "complete" << endl;

}

void SimpleFilter::nppi_example(Frame *vp)
{
    cout << "processGPU" << endl;
    int width = vp->width;
    int height = vp->height;
    int frame_size = width * height;
    int pStepBytes_dSrc8uC1;
    int pStepBytes_dSrc32fC1;
    Npp64f *pSum;
    int nBufferSize;
    Npp64f nHostSum;

    try {
        NppStreamContext ctx = initializeNppStreamContext();
        eh.ck(nppiSumGetBufferHostSize_8u_C1R({width, height}, &nBufferSize));
        cout << "nBufferSize: " << nBufferSize << endl;
        Npp8u *pDeviceBuffer;
        cudaMalloc((void**)(&pDeviceBuffer), nBufferSize);
        cudaMalloc((void**)(&pSum), sizeof(Npp64f));

        Npp8u *dSrc8uC1 = nppiMalloc_8u_C1(width, height, &pStepBytes_dSrc8uC1);
        cout << "pStepBytes_dSrc8uC1: " << pStepBytes_dSrc8uC1 << endl;
        eh.ck(cudaMemcpy(dSrc8uC1, vp->frame->data[0], sizeof(Npp8u) * frame_size, cudaMemcpyHostToDevice), "copy frame data to device");
        eh.ck(nppiSum_8u_C1R_Ctx(dSrc8uC1, pStepBytes_dSrc8uC1, {width, height}, pDeviceBuffer, pSum, ctx));

        eh.ck(cudaMemcpy(&nHostSum, pSum, sizeof(Npp64f), cudaMemcpyDeviceToHost), "retrieve results of sum");
        cout << "nHostSum: " << nHostSum << endl;

        Npp32f *dSrc32fC1 = nppiMalloc_32f_C1(width, height, &pStepBytes_dSrc32fC1);
        cout << "pStepBytes_dSrc32fC1: " << pStepBytes_dSrc32fC1 << endl;
        eh.ck(nppsConvert_8u32f_Ctx(dSrc8uC1, dSrc32fC1, frame_size, ctx), "convert frame data to float");

        nppiFree(dSrc8uC1);
        nppiFree(dSrc32fC1);
        cudaFree(pDeviceBuffer);
        cudaFree(pSum);
    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }
}

void SimpleFilter::cuda_example(Frame *vp)
{
    int width = vp->width;
    int height = vp->height;
    int frame_size = width * height;

    Npp8u *pDvcSrcUint;
    Npp32f *pDvcSrcFloat;
    Npp8u * pDeviceBuffer;
    Npp32f *pSum;
    Npp32f nSumHost;

    int nBufferSize;


    cout << "start" << endl;

    try {
        eh.ck(nppsSumGetBufferSize_32f(frame_size, &nBufferSize), "Determine scratch buffer size needed for nppsSum");
        eh.ck(cudaMalloc((void**)(& pDvcSrcUint), sizeof(Npp8u) * frame_size), "Allocate device source buffer");
        eh.ck(cudaMemcpy(pDvcSrcUint, vp->frame->data[0], sizeof(Npp8u) * frame_size, cudaMemcpyHostToDevice), "Copy frame picture data buffer to device");
        eh.ck(cudaMalloc((void **)(&pDvcSrcFloat), sizeof(Npp32f) * frame_size), "Allocate a float buffer version of source for device");
        eh.ck(nppsConvert_8u32f(pDvcSrcUint, pDvcSrcFloat, frame_size), "Convert frame date to float");
        eh.ck(cudaMalloc((void **)(&pDeviceBuffer), nBufferSize), "Allocate the scratch buffer for nppsSum");
        eh.ck(cudaMalloc((void **)(&pSum), sizeof(Npp32f)), "Allocate a buffer for the result of nppsSum");
        eh.ck(nppsSum_32f(pDvcSrcFloat, frame_size, pSum, pDeviceBuffer), "nppsSum");
        eh.ck(cudaMemcpy(&nSumHost, pSum, sizeof(Npp32f), cudaMemcpyDeviceToHost), "Copy the result of nppsSum to host");

        cout << "sum 1 = " << nSumHost << endl;

    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }

    cout << "first task complete" << endl;

    try {
        const Npp32f factor = 255.0f;
        eh.ck(nppsDivC_32f_I(factor, pDvcSrcFloat, frame_size), "normalize frame data");
        eh.ck(nppsSum_32f(pDvcSrcFloat, frame_size, pSum, pDeviceBuffer), "nppsSum");
        eh.ck(cudaMemcpy(&nSumHost, pSum, sizeof(Npp32f), cudaMemcpyDeviceToHost), "Copy the result of nppsSum to host");

        cout << "sum 2 = " << nSumHost << endl;

    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }

    eh.ck(cudaFree(pDvcSrcFloat));;
    eh.ck(cudaFree(pDeviceBuffer));
    eh.ck(cudaFree(pDvcSrcUint));
    cout << "complete" << endl;

}

NppStreamContext SimpleFilter::initializeNppStreamContext()
{
    NppStreamContext npp_ctx;
    try {
        int device;
        eh.ck(cudaGetDevice(&device), "get current device");
        cout << "device: " << device << endl;
        cudaDeviceProp prop;
        eh.ck(cudaGetDeviceProperties(&prop, device), "get device props");
        cout << "name: " << prop.name << endl;
        cout << "total mem: " << prop.totalGlobalMem << endl;
        cout << "clock: " << prop.clockRate << endl;

        int major;
        int minor;
        eh.ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "get major compute capability");
        eh.ck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "get minor compute capability");

        npp_ctx.nCudaDeviceId = device;
        npp_ctx.nMultiProcessorCount = prop.multiProcessorCount;
        npp_ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        npp_ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
        npp_ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
        npp_ctx.nCudaDevAttrComputeCapabilityMajor = major;
        npp_ctx.nCudaDevAttrComputeCapabilityMinor = minor;

        cudaStream_t pStream;
        eh.ck(cudaStreamCreate(&pStream), "create cuda stream");
        unsigned int flags;
        eh.ck(cudaStreamGetFlags(pStream, &flags), "get cuda stream flags");

        npp_ctx.hStream = pStream;
        npp_ctx.nStreamFlags = flags;


    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }
    return npp_ctx;
}

void SimpleFilter::test()
{
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
    if (MW->model == nullptr) {
        MW->model = new Model(MW);
        MW->model->initialize(MW->cfg_file, MW->weights_file, MW->names_file, 0);
    }

    vector<bbox_t> result = MW->model->infer(vp, 0.2);
    for (int i = 0; i < result.size(); i++) {
        QRect rect(result[i].x, result[i].y, result[i].w, result[i].h);
        YUVColor green(Qt::green);
        vp->drawBox(rect, 1, green);
    }
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

/*
try {
    string in_filename = "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.3/common/data/Lena.pgm";
    string out_filename = "C:/Users/sr996/data/Lena_box_filter.pgm";
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(in_filename, oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    NppiSize oMaskSize = {5, 5};
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    Npp32f *pDvcSrcFloat;
    int nBufferSize;
    Npp8u *pScratchBuffer;
    Npp32f *pSum;
    Npp32f nSumHost;
    int frame_size = oDeviceSrc.width() * oDeviceSrc.height();

    eh.ck(cudaMalloc(&pDvcSrcFloat, sizeof(Npp32f) * frame_size));
    eh.ck(nppsConvert_8u32f(oDeviceSrc.data(), pDvcSrcFloat, frame_size));
    eh.ck(nppsSumGetBufferSize_32f(frame_size, &nBufferSize));
    eh.ck(cudaMalloc((void**)(&pScratchBuffer), nBufferSize));
    eh.ck(cudaMalloc((void**)(&pSum), sizeof(Npp32f)));
    eh.ck(nppsSum_32f(pDvcSrcFloat, frame_size, pSum, pScratchBuffer));
    eh.ck(cudaMemcpy(&nSumHost, pSum, sizeof(Npp32f), cudaMemcpyDeviceToHost));

    cout << "sum = " << nSumHost << endl;

    Npp32f factor = 255.0f;
    Npp32f *normalized;
    NppiSize sizeRoi {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    eh.ck(cudaMalloc((void**)(&normalized), sizeof(Npp32f) * frame_size));
    eh.ck(nppiDivC_32f_C1R(pDvcSrcFloat, 512*512, factor, normalized, 512*512, {512, 512}), "Normalize the data");
    eh.ck(cudaFree(normalized), "Free data structure");

    eh.ck(nppiFilterBoxBorder_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                     oSrcSize, oSrcOffset,
                                     oDeviceDst.data(), oDeviceDst.pitch(),
                                     oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE));

    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    saveImage(out_filename, oHostDst);

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}
catch (const exception& e) {
    cout << e.what() << endl;
}
*/
