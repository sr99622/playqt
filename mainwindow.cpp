#include "mainwindow.h"
#include "Ffplay/cmdutils.h"

using namespace cv::cuda;

MainWindow::MainWindow(CommandOptions *co, QWidget *parent) : QMainWindow(parent)
{
    this->co = co;
    filename = QString(co->input_filename);
    av_log_set_level(AV_LOG_PANIC);

    QScreen *screen = QApplication::primaryScreen();
    QRect screenSize = screen->geometry();
    int aw = 1300;
    int ah = 800;
    int cx = screenSize.center().x();
    int cy = screenSize.center().y();
    int ax = cx - aw/2;
    int ay = cy - ah/2;

    setWindowTitle(filename);
    settings = new QSettings("PlayQt", "Program Settings");

    viewerDialog = new ViewerDialog(this);

    splitter = new QSplitter(Qt::Orientation::Horizontal, this);
    tabWidget = new QTabWidget(this);
    tabWidget->setTabPosition(QTabWidget::East);
    tabWidget->setMinimumWidth(100);
    videoPanel = new FilePanel(this);
    videoPanel->setDirectory(QStandardPaths::writableLocation(QStandardPaths::MoviesLocation));
    picturePanel = new FilePanel(this);
    picturePanel->setDirectory(QStandardPaths::writableLocation(QStandardPaths::PicturesLocation));
    audioPanel = new FilePanel(this);
    audioPanel->setDirectory(QStandardPaths::writableLocation(QStandardPaths::MusicLocation));
    cameraPanel = new CameraPanel(this);
    streamPanel = new StreamPanel(this);
    tabWidget->addTab(videoPanel, "Videos");
    tabWidget->addTab(picturePanel, "Pictures");
    tabWidget->addTab(audioPanel, "Audio");
    tabWidget->addTab(cameraPanel, "Cameras");
    tabWidget->addTab(streamPanel, "Streams");
    mainPanel = new MainPanel(this);
    splitter->addWidget(mainPanel);
    splitter->addWidget(tabWidget);
    setCentralWidget(splitter);

    names_file = settings->value("MainWindow/names_file", "").toString();
    cfg_file = settings->value("MainWindow/cfg_file", "").toString();
    weights_file = settings->value("MainWindow/weights_file", "").toString();
    initializeModelOnStartup = settings->value("MainWindow/initializeModelOnStartup", false).toBool();

    if (initializeModelOnStartup) {
        model = new Model(this);
        model->initialize(cfg_file, weights_file, names_file, 0);
    }

    messageBox = new MessageBox(this);
    modelConfigureDialog = new ModelConfigureDialog(this);
    filterDialog = new FilterDialog(this);
    optionDialog = new OptionDialog(this);
    parameterDialog = new ParameterDialog(this);

    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction("&Next");
    fileMenu->addAction("&Previous");
    fileMenu->addAction("&Save");
    fileMenu->addAction("&Quit");

    QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
    toolsMenu->addAction("&Model Options");
    toolsMenu->addAction("&Filters");
    toolsMenu->addAction("&Set Parameters");
    toolsMenu->addAction("M&essages");

    QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction("&Options");

    connect(fileMenu, SIGNAL(triggered(QAction*)), this, SLOT(fileMenuAction(QAction*)));
    connect(toolsMenu, SIGNAL(triggered(QAction*)), this, SLOT(toolsMenuAction(QAction*)));
    connect(helpMenu, SIGNAL(triggered(QAction*)), this, SLOT(helpMenuAction(QAction*)));
    //connect(co, SIGNAL(showHelp(const QString&)), this, SLOT(showHelp(const QString&)));
    connect(co, SIGNAL(showHelp(const QString&)), optionDialog->panel, SLOT(showConfig(const QString&)));

    move(ax, ay);


    //test();
}

MainWindow::~MainWindow()
{

}

void MainWindow::runLoop()
{
    av_init_packet(&flush_pkt);
    flush_pkt.data = (uint8_t*)&flush_pkt;

    is = VideoState::stream_open(/*this, */co->input_filename, NULL, co, &display);
    is->mainWindow = this;
    is->filter = new SimpleFilter(this);
    is->flush_pkt = &flush_pkt;

    e.event_loop(is);
    if (is) {
        is->stream_close();
    }
}

void MainWindow::initializeSDL()
{
    display.window = SDL_CreateWindowFrom((void*)mainPanel->displayContainer->display->winId());
    display.renderer = SDL_CreateRenderer(display.window, -1, 0);
    SDL_GetRendererInfo(display.renderer, &display.renderer_info);

    if (!display.window || !display.renderer || !display.renderer_info.num_texture_formats) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create window or renderer: %s", SDL_GetError());
    }
}

void MainWindow::resizeEvent(QResizeEvent *event)
{

}

void MainWindow::moveEvent(QMoveEvent *event)
{

}

void MainWindow::closeEvent(QCloseEvent *event)
{
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
        cout << "SDL_PushEvent Failure: " << SDL_GetError() << endl;

    QMainWindow::closeEvent(event);
}

void MainWindow::msg(const QString &str)
{
    messageBox->message->append(str);
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
    else if (action->text() == "&Filters")
        filterDialog->show();
    else if (action->text() == "&Set Parameters")
        parameterDialog->show();
    else if (action->text() == "M&essages")
        messageBox->show();
}

void MainWindow::helpMenuAction(QAction *action)
{
    if (action->text() == "&Options")
        optionDialog->show();
}

void MainWindow::getNames(QString names_file)
{
    this->names_file = names_file;
    ifstream file(names_file.toLatin1().data());
    if (!file.is_open())
        return;

    for (string line; getline(file, line);)
        obj_names.push_back(line);
}

void MainWindow::showHelp(const QString &str)
{
    cout << str.toStdString();
}

void MainWindow::test()
{
    viewerDialog->show();
    //co->video_codec_name = "h264_qsv";
    //cout << "video_codec_name: " << co->video_codec_name << endl;
}

