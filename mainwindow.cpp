#include "mainwindow.h"
#include "Ffplay/cmdutils.h"

using namespace cv::cuda;

MainWindow::MainWindow(CommandOptions *co, QWidget *parent) : QMainWindow(parent)
{
    this->co = co;
    filename = QString(co->input_filename);
    av_log_set_level(AV_LOG_PANIC);

    screen = QApplication::primaryScreen();
    QRect screenSize = screen->geometry();
    int aw = 1300;
    int ah = 800;
    int cx = screenSize.center().x();
    int cy = screenSize.center().y();
    int ax = cx - aw/2;
    int ay = cy - ah/2;

    setWindowTitle(filename);
    settings = new QSettings("PlayQt", "Program Settings");
    status = new QStatusBar(this);
    setStatusBar(status);

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(poll()));
    timer->start(1000 / 30);

    QFile f("C:/Users/sr996/Projects/playqt/darkstyle.qss");
    QString style;

    if (!f.exists()) {
        cout << "Error: MainWindow::getThemes() Style sheet not found" << endl;
    }
    else {
        f.open(QFile::ReadOnly | QFile::Text);
        style = QString(f.readAll());
        style.replace("background_dark", "#283445");
        style.replace("background_medium", "#3E4754");
        style.replace("background_light", "#566170");
        style.replace("foreground_light", "#C6D9F2");
        style.replace("selection_light", "#FFFFFF");
        setStyleSheet(style);
    }

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

    messageBox = new MessageBox(this);
    filterDialog = new FilterDialog(this);
    filterDialog->panel->restoreSettings(settings);
    optionDialog = new OptionDialog(this);
    parameterDialog = new ParameterDialog(this);
    parameterDialog->panel->restoreSettings(settings);
    filterChain = new FilterChain(this);

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

    sdlCustomEventType = SDL_RegisterEvents(1);
    av_init_packet(&flush_pkt);
    flush_pkt.data = (uint8_t*)&flush_pkt;
    e = new EventHandler(this);

    viewerDialog = new ViewerDialog(this);
    //test();
}

MainWindow::~MainWindow()
{

}

void MainWindow::runLoop()
{
    if (co->input_filename) {
        is = VideoState::stream_open(this);
        is->filter = new SimpleFilter(this);

        timer->start();
        e->event_loop();
    }
}

void MainWindow::poll()
{
    if (is != nullptr) {

        if (is->paused /* && mainPanel->controlPanel->engageFilter->isChecked() */ ) {
            guiUpdate(0);
        }
        //else {
            double remaining_time = REFRESH_RATE;
            if (is->show_mode != SHOW_MODE_NONE && (!is->paused || is->force_refresh))
                is->video_refresh(&remaining_time);
        //}
    }
}

void MainWindow::guiUpdate(int arg)
{
    if (arg > 0)
        cout << "MainWindow::guiUpdate: " << arg << endl;

    if (is != nullptr) {
        is->video_display();
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

void MainWindow::paintEvent(QPaintEvent *event)
{
    /*
    cout << "MainWindow::paintEvent: " << TS << endl;
    if (is) {
        if (is->paused) {
            is->video_display();
            cout << "UPDATE" << endl;
        }
    }
    else {
        cout << "WTF" << endl;
    }
    */
    QMainWindow::paintEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    //cout << QTime::currentTime().toString("hh:mm:ss.zzz").toStdString() << endl;
    //if (is != nullptr) {
    //    is->video_display();
    //}
    QMainWindow::resizeEvent(event);
}

void MainWindow::moveEvent(QMoveEvent *event)
{
    //if (is != nullptr) {
    //    is->video_display();
    //}
    QMainWindow::moveEvent(event);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    settings->setValue("MainWindow/geometry", saveGeometry());

    filterDialog->panel->saveSettings(settings);
    parameterDialog->panel->saveSettings(settings);

    SDL_Event sdl_event;
    sdl_event.type = FF_QUIT_EVENT;
    SDL_PushEvent(&sdl_event);

    //timer->stop();

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
        cout << "&ModelOptions" << endl;
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
    /*
    this->names_file = names_file;
    ifstream file(names_file.toLatin1().data());
    if (!file.is_open())
        return;

    for (string line; getline(file, line);)
        obj_names.push_back(line);
    */
}

void MainWindow::showHelp(const QString &str)
{
    cout << str.toStdString();
}

void MainWindow::ping(const vector<bbox_t>* arg)
{
    cout << "ping: " << arg->size() << endl;
    for (const bbox_t detection : *arg) {
        cout << " "  << detection.track_id;
    }
    cout << endl;
}

void MainWindow::test()
{
    cout << "MainWindow::test" << endl;

    QString str = "60000000 ";
    bool ok;
    double arg = str.toDouble(&ok);
    if (ok) {
        cout << "result ok: " << arg << endl;
    }
    else {
        cout << "NOT OK" << endl;
    }


    /*
    vector<OptionDef> *saved_options = &parameterDialog->panel->saved_options;
    cout << "number of saved options: " << saved_options->size() << endl;
    for (size_t i = 0; i < saved_options->size(); i++) {
        OptionDef *opt = &saved_options->at(i);
        cout << "option name: " << opt->name << endl;
        if (opt->flags & OPT_STRING) {
            char * str = *(char**)opt->u.dst_ptr;
            cout << "arg: " << str << endl;
        }
        else if (opt->name == "vf") {
            for (int j = 0; j < co->nb_vfilters; j++) {
                cout << " " << co->vfilters_list[j];
            }
            cout << endl;
        }
    }
    */
}

