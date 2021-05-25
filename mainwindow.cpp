#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    av_log_set_level(AV_LOG_PANIC);
    mainPanel = new MainPanel(this);
    model = new Model(this);
    setCentralWidget(mainPanel);

    QString names_file = "C:/Users/sr996/models/reduced/ami1/coco.names";
    QString cfg_file = "C:/Users/sr996/models/reduced/ami1/yolov4.cfg";
    QString weights_file = "C:/Users/sr996/models/reduced/ami1/yolov4_final.weights";
    int gpu_id = 0;
    model->initialize(cfg_file, weights_file, names_file, gpu_id);

    setWindowTitle("Play Qt");
    test();
}

MainWindow::~MainWindow()
{

}

void MainWindow::runLoop()
{
    av_init_packet(&flush_pkt);
    flush_pkt.data = (uint8_t*)&flush_pkt;

    is = VideoState::stream_open(co->input_filename, NULL, co, &display);
    is->filter = new SimpleFilter(this);
    is->flush_pkt = &flush_pkt;

    e.event_loop(is);
    if (is) {
        is->stream_close();
    }
}

void MainWindow::initializeSDL()
{
    display.window = SDL_CreateWindowFrom((void*)mainPanel->label->winId());
    display.renderer = SDL_CreateRenderer(display.window, -1, 0);
    SDL_GetRendererInfo(display.renderer, &display.renderer_info);

    if (!display.window || !display.renderer || !display.renderer_info.num_texture_formats) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create window or renderer: %s", SDL_GetError());
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    SDL_Event sdl_event;
    sdl_event.type = SDL_KEYDOWN;
    sdl_event.key.keysym.sym = SDLK_ESCAPE;
    int result = SDL_PushEvent(&sdl_event);
    if (result < 0)
        cout << "SDL_PushEvent Failure" << SDL_GetError() << endl;
    QMainWindow::closeEvent(event);
}

void MainWindow::test()
{
    cout << "test 1" << endl;
    cudaMallocManaged(&ptr_image, 1024 * sizeof(uint8_t));
    cudaMallocManaged(&ptr_data, 1024 * sizeof(float));
    cudaMemset(ptr_image, 128, 1024 * sizeof(uint8_t));
    cudaMemset(ptr_data, 0, 1024 * sizeof(float));
    cudaFree(ptr_image);
    cudaFree(ptr_data);
    cout << "test 2" << endl;

    Npp32f * pSrc;
    Npp32f * pSum;
    Npp8u * pDeviceBuffer;
    int nLength = 1024;

    // Allocate the device memroy.
    cudaMalloc((void **)(&pSrc), sizeof(Npp32f) * nLength);
    nppsSet_32f(1.0f, pSrc, nLength);
    cudaMalloc((void **)(&pSum), sizeof(Npp32f) * 1);

    // Compute the appropriate size of the scratch-memory buffer
    int nBufferSize;
    nppsSumGetBufferSize_32f(nLength, &nBufferSize);
    // Allocate the scratch buffer
    cudaMalloc((void **)(&pDeviceBuffer), nBufferSize);

    // Call the primitive with the scratch buffer
    nppsSum_32f(pSrc, nLength, pSum, pDeviceBuffer);
    Npp32f nSumHost;
    cudaMemcpy(&nSumHost, pSum, sizeof(Npp32f) * 1, cudaMemcpyDeviceToHost);
    cout << "sum = " << nSumHost << endl;

    // Free the device memory
    cudaFree(pSrc);
    cudaFree(pDeviceBuffer);
    cudaFree(pSum);

    QString filename = "C:/Users/sr996/Pictures/20210502091315.jpg";
    cout << "filename: " << filename.toStdString() << endl;
    QPixmap pixmap(filename);
    cout << "width: " << pixmap.size().width() << " height: " << pixmap.size().height() << endl;
}

