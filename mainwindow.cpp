#include "mainwindow.h"
#include "Ffplay/cmdutils.h"

using namespace cv::cuda;

MainWindow::MainWindow(CommandOptions *co, QWidget *parent) : QMainWindow(parent)
{
    this->co = co;
    co->mainWindow = this;
    filename = QString(co->input_filename);
    av_log_set_level(AV_LOG_PANIC);


    screen = QApplication::primaryScreen();
    settings = new QSettings("PlayQt", "Program Settings");

    QRect screenSize = screen->geometry();
    int w = min(APP_DEFAULT_WIDTH, screenSize.width());
    int h = min(APP_DEFAULT_HEIGHT, screenSize.height());
    int x = screenSize.center().x() - w/2;
    int y = screenSize.center().y() - h/2;
    QRect defaultGeometry(x, y, w, h);

    if (settings->contains(geometryKey))
        restoreGeometry(settings->value(geometryKey).toByteArray());
    else
        setGeometry(defaultGeometry);

    QString title = "PlayQt";
    if (co->input_filename) {
        if (!strcmp(co->input_filename, "-")) {
            title = title + " - pipe:";
        }
        else {
            QFileInfo fi(co->input_filename);
            title = title + " - " + fi.fileName();
        }
    }

    setWindowTitle(title);
    status = new QStatusBar(this);
    setStatusBar(status);

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(poll()));
    timer->start(1000 / 30);

    QFile f(":darkstyle.qss");
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
        style.replace("selection_dark", "#2C5059");
        style.replace("selection_medium", "#4A8391");
        setStyleSheet(style);
    }


    tabWidget = new QTabWidget(this);
    tabWidget->setTabPosition(QTabWidget::East);
    tabWidget->setMinimumWidth(100);

    videoPanel = new FilePanel(this);
    if (settings->contains(videoPanelHeaderKey))
        videoPanel->tree->header()->restoreState(settings->value(videoPanelHeaderKey).toByteArray());
    if (settings->contains(videoPanelDirKey))
        videoPanel->setDirectory(settings->value(videoPanelDirKey).toString());
    else
        videoPanel->setDirectory(QStandardPaths::writableLocation(QStandardPaths::MoviesLocation));

    picturePanel = new FilePanel(this);
    if (settings->contains(picturePanelHeaderKey))
        picturePanel->tree->header()->restoreState(settings->value(picturePanelHeaderKey).toByteArray());
    if (settings->contains(picturePanelDirKey))
        picturePanel->setDirectory(settings->value(picturePanelDirKey).toString());
    else
        picturePanel->setDirectory(QStandardPaths::writableLocation(QStandardPaths::PicturesLocation));

    audioPanel = new FilePanel(this);
    if (settings->contains(audioPanelHeaderKey))
        audioPanel->tree->header()->restoreState(settings->value(audioPanelHeaderKey).toByteArray());
    if (settings->contains(audioPanelDirKey))
        audioPanel->setDirectory(settings->value(audioPanelDirKey).toString());
    else
        audioPanel->setDirectory(QStandardPaths::writableLocation(QStandardPaths::MusicLocation));

    cameraPanel = new CameraPanel(this);
    streamPanel = new StreamPanel(this);
    tabWidget->addTab(videoPanel, tr("Videos"));
    tabWidget->addTab(picturePanel, tr("Pictures"));
    tabWidget->addTab(audioPanel, tr("Audio"));
    tabWidget->addTab(cameraPanel, tr("Cameras"));
    tabWidget->addTab(streamPanel, tr("Streams"));
    mainPanel = new MainPanel(this);

    splitter = new QSplitter(Qt::Orientation::Horizontal, this);
    splitter->addWidget(mainPanel);
    splitter->addWidget(tabWidget);
    if (settings->contains(splitterKey))
        splitter->restoreState(settings->value(splitterKey).toByteArray());

    setCentralWidget(splitter);

    messageBox = new MessageBox(this);
    filterDialog = new FilterDialog(this);
    filterDialog->panel->restoreSettings(settings);
    optionDialog = new OptionDialog(this);
    parameterDialog = new ParameterDialog(this);
    parameterDialog->panel->restoreSettings(settings);
    filterChain = new FilterChain(this);

    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    QAction *actOpen = new QAction(tr("&Open"));
    actOpen->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_O));
    fileMenu->addAction(actOpen);
    QAction *actExit = new QAction(tr("E&xit"));
    actExit->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_X));
    fileMenu->addAction(actExit);

    QMenu *mediaMenu = menuBar()->addMenu(tr("&Media"));
    QAction *actPlay = new QAction(tr("&Play/Pause"));
    actPlay->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_P));
    mediaMenu->addAction(actPlay);
    QAction *actRewind = new QAction(tr("&Rewind"));
    actRewind->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
    mediaMenu->addAction(actRewind);
    QAction *actFastForward = new QAction(tr("&Fast Forward"));
    actFastForward->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F));
    mediaMenu->addAction(actFastForward);
    QAction *actMute = new QAction(tr("&Mute"));
    actMute->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_M));
    mediaMenu->addAction(actMute);
    QAction *actQuit = new QAction(tr("&Quit"));
    actQuit->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q));
    mediaMenu->addAction(actQuit);

    QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
    toolsMenu->addAction(tr("&Filters"));
    toolsMenu->addAction(tr("Set &Parameters"));
    toolsMenu->addAction(tr("&Messages"));

    QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(tr("&Options"));

    /*
    QShortcut *ctrl_F = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_F), this);
    connect(ctrl_F, SIGNAL(activated()), filterDialog, SLOT(show()));
    QShortcut *ctrl_P = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_P), this);
    connect(ctrl_P, SIGNAL(activated()), parameterDialog, SLOT(show()));
    */

    connect(fileMenu, SIGNAL(triggered(QAction*)), this, SLOT(fileMenuAction(QAction*)));
    connect(mediaMenu, SIGNAL(triggered(QAction*)), this, SLOT(mediaMenuAction(QAction*)));
    connect(toolsMenu, SIGNAL(triggered(QAction*)), this, SLOT(toolsMenuAction(QAction*)));
    connect(helpMenu, SIGNAL(triggered(QAction*)), this, SLOT(helpMenuAction(QAction*)));
    connect(co, SIGNAL(showHelp(const QString&)), optionDialog->panel, SLOT(showConfig(const QString&)));

    sdlCustomEventType = SDL_RegisterEvents(1);
    av_init_packet(&flush_pkt);
    flush_pkt.data = (uint8_t*)&flush_pkt;
    e = new EventHandler(this);
    timer->start();

    //viewerDialog = new ViewerDialog(this);

    if (co->input_filename) {
        launcher = new Launcher(this);
        connect(launcher, SIGNAL(done()), mainPanel->controlPanel, SLOT(play()));
        QThreadPool::globalInstance()->tryStart(launcher);
    }


}

MainWindow::~MainWindow()
{
    if (quitter)
        delete quitter;
}

void MainWindow::runLoop()
{
    if (is) {
        if (!quitter) {
            quitter = new Quitter(this);
            connect(quitter, SIGNAL(done()), mainPanel->controlPanel, SLOT(play()));
        }
        QThreadPool::globalInstance()->tryStart(quitter);
    }
    else {
        start();
    }
}

void MainWindow::start()
{
    cout << "start: " << co->input_filename << endl;
    if (co->input_filename) {
        is = VideoState::stream_open(this);
        e->event_loop();
    }
}

void MainWindow::poll()
{
    if (is != nullptr) {
        if (is->paused) {
            guiUpdate(0);
        }

        double remaining_time = REFRESH_RATE;
        if (is->show_mode != SHOW_MODE_NONE && (!is->paused || is->force_refresh))
            is->video_refresh(&remaining_time);
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
    //cout << "mainwindow paint event" << TS << endl;
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

void MainWindow::showEvent(QShowEvent *event)
{
    QMainWindow::showEvent(event);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    settings->setValue(geometryKey, saveGeometry());
    settings->setValue(splitterKey, splitter->saveState());
    settings->setValue(videoPanelHeaderKey, videoPanel->tree->header()->saveState());
    settings->setValue(videoPanelDirKey, videoPanel->directorySetter->directory);
    settings->setValue(picturePanelHeaderKey, picturePanel->tree->header()->saveState());
    settings->setValue(picturePanelDirKey, picturePanel->directorySetter->directory);
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
    if (action->text() == "&Open") {
        QString default_path = QDir::homePath();

        QString path = QFileDialog::getOpenFileName(this, "", default_path, "");
        if (path.length() > 0) {
            co->input_filename = av_strdup(path.toLatin1().data());
            mainPanel->controlPanel->play();
        }

    }
    else if (action->text() == "E&xit") {
        close();
    }
}

void MainWindow::mediaMenuAction(QAction *action)
{
    cout << action->text().toStdString() << endl;
    if (action->text() == "&Play/Pause")
        mainPanel->controlPanel->play();
    else if (action->text() == "&Rewind")
        mainPanel->controlPanel->rewind();
    else if (action->text() == "&Fast Forward")
        mainPanel->controlPanel->fastforward();
    else if (action->text() == "&Quit")
        mainPanel->controlPanel->quit();
    else if (action->text() == "&Mute")
        mainPanel->controlPanel->mute();
}

void MainWindow::toolsMenuAction(QAction *action)
{
    cout << action->text().toStdString() << endl;
    if (action->text() == "&Filters")
        filterDialog->show();
    else if (action->text() == "Set &Parameters")
        parameterDialog->show();
    else if (action->text() == "&Messages")
        messageBox->show();
}

void MainWindow::helpMenuAction(QAction *action)
{
    if (action->text() == "&Options")
        optionDialog->show();
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
}

Launcher::Launcher(QMainWindow *parent)
{
    mainWindow = parent;
}

void Launcher::run()
{
    while (!mainWindow->isVisible())
        QThread::msleep(10);

    emit done();
}

Quitter::Quitter(QMainWindow *parent)
{
    mainWindow = parent;
    setAutoDelete(false);
}

void Quitter::run()
{
    if (MW->is)
        MW->is->abort_request = 1;

    SDL_Event event;
    event.type = FF_QUIT_EVENT;
    event.user.data1 = this;
    SDL_PushEvent(&event);

    while (MW->e->running)
        QThread::msleep(10);

    MW->mainPanel->controlPanel->stopped = true;
    MW->mainPanel->controlPanel->paused = false;

    emit done();
}
