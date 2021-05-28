#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    av_log_set_level(AV_LOG_PANIC);

    setWindowTitle("Play Qt");
    settings = new QSettings("PlayQt", "Program Settings");

    mainPanel = new MainPanel(this);
    model = new Model(this);
    setCentralWidget(mainPanel);

    names_file = settings->value("MainWindow/names_file", "").toString();
    cfg_file = settings->value("MainWindow/cfg_file", "").toString();
    weights_file = settings->value("MainWindow/weights_file", "").toString();
    initializeModelOnStartup = settings->value("MainWindow/initializeModelOnStartup", false).toBool();

    if (initializeModelOnStartup)
        model->initialize(cfg_file, weights_file, names_file, 0);

    cout << "open event: " << names_file.toStdString() << endl;

    modelConfigureDialog = new ModelConfigureDialog(this);

    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction("&Next");
    fileMenu->addAction("&Previous");
    fileMenu->addAction("&Save");
    fileMenu->addAction("&Quit");

    QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
    toolsMenu->addAction("&Model Options");
    toolsMenu->addAction("&File Filters");
    connect(fileMenu, SIGNAL(triggered(QAction*)), this, SLOT(fileMenuAction(QAction*)));
    connect(toolsMenu, SIGNAL(triggered(QAction*)), this, SLOT(toolsMenuAction(QAction*)));

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
    cout << "close event: " << names_file.toStdString() << endl;

    settings->setValue("MainWindow/geometry", saveGeometry());
    settings->setValue("MainWindow/names_file",names_file);
    settings->setValue("MainWindow/cfg_file", cfg_file);
    settings->setValue("MainWindow/weights_file", weights_file);
    settings->setValue("MainWindow/initializeModelOnStartup", initializeModelOnStartup);

    SDL_Event sdl_event;
    sdl_event.type = SDL_KEYDOWN;
    sdl_event.key.keysym.sym = SDLK_ESCAPE;
    int result = SDL_PushEvent(&sdl_event);
    if (result < 0)
        cout << "SDL_PushEvent Failure" << SDL_GetError() << endl;

    QMainWindow::closeEvent(event);
}

void MainWindow::fileMenuAction(QAction *action)
{
    cout << action->text().toStdString() << endl;
}

void MainWindow::toolsMenuAction(QAction *action)
{
    cout << action->text().toStdString() << endl;
    if (action->text() == "&Model Options")
        modelConfigureDialog->show();
}

void MainWindow::get_names(QString names_file)
{
    this->names_file = names_file;
    ifstream file(names_file.toLatin1().data());
    if (!file.is_open())
        return;

    for (string line; getline(file, line);)
        obj_names.push_back(line);
}

void MainWindow::test()
{
    cout << "MainWindow::test" << endl;
    //is->filter->test();
}

