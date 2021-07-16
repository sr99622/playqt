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

    if (!f.exists()) {
        msg("Error: MainWindow::getThemes() Style sheet not found");
    }
    else {
        f.open(QFile::ReadOnly | QFile::Text);
        style = QString(f.readAll());
        style.replace("background_light",  "#566170");
        style.replace("background_medium", "#3E4754");
        style.replace("background_dark",   "#283445");
        style.replace("foreground_light",  "#C6D9F2");
        style.replace("foreground_medium", "#9DADC2");
        style.replace("foreground_dark",   "#808D9E");
        style.replace("selection_light",   "#FFFFFF");
        style.replace("selection_medium",  "#4A8391");
        style.replace("selection_dark",    "#2C5059");
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
    //filterDialog->panel->restoreSettings(settings);
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
    QAction *actPlay = new QAction(QIcon(":play.png"), tr("&Play/Pause"));
    actPlay->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_P));
    mediaMenu->addAction(actPlay);
    QAction *actRewind = new QAction(QIcon(":rewind"), tr("&Rewind"));
    actRewind->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
    mediaMenu->addAction(actRewind);
    QAction *actFastForward = new QAction(QIcon(":fast-forward"), tr("Fas&t Forward"));
    actFastForward->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_T));
    mediaMenu->addAction(actFastForward);
    QAction *actPrevious = new QAction(QIcon(":previous"), tr("Pre&vious"));
    actPrevious->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_V));
    mediaMenu->addAction(actPrevious);
    QAction *actNext = new QAction(QIcon(":next"), tr("&Next"));
    actNext->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_N));
    mediaMenu->addAction(actNext);
    QAction *actMute = new QAction(QIcon(":mute"), tr("&Mute"));
    actMute->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_M));
    mediaMenu->addAction(actMute);
    QAction *actQuit = new QAction(QIcon(":stop"), tr("&Quit"));
    actQuit->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Q));
    mediaMenu->addAction(actQuit);

    QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
    QAction *actFilter = new QAction(tr("&Filters"));
    actFilter->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F));
    toolsMenu->addAction(actFilter);
    QAction *actEngage = new QAction(tr("&Engage"));
    actEngage->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_E));
    toolsMenu->addAction(actEngage);
    QAction *actSetParameters = new QAction(tr("&Set Parameters"));
    actSetParameters->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_S));
    toolsMenu->addAction(actSetParameters);
    QAction *actMessages = new QAction(tr("Messa&ges"));
    actMessages->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_G));
    toolsMenu->addAction(actMessages);

    QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(tr("&Options"));

    connect(fileMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(mediaMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(toolsMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(helpMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(co, SIGNAL(showHelp(const QString&)), optionDialog->panel, SLOT(showConfig(const QString&)));

    sdlCustomEventType = SDL_RegisterEvents(1);
    av_init_packet(&flush_pkt);
    flush_pkt.data = (uint8_t*)&flush_pkt;
    e = new EventHandler(this);
    timer->start();

    //viewerDialog = new ViewerDialog(this);

    int startup_volume = av_clip(co->startup_volume, 0, 128);
    if (startup_volume == 0)
        mainPanel->controlPanel->mute();
    else
        mainPanel->controlPanel->volumeSlider->setValue(startup_volume);

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
        if (tabWidget->tabText(tabWidget->currentIndex()) == "Cameras") {
            CameraPanel *cameraPanel = (CameraPanel*)tabWidget->currentWidget();
            Camera *camera = cameraPanel->cameraList->getCurrentCamera();
            if (camera) {
                QString title = "PlayQt - ";
                title += camera->onvif_data->camera_name;
                setWindowTitle(title);
            }
        }
        else {
            QFileInfo fi(co->input_filename);
            QString title = "PlayQt - " + fi.fileName();
            setWindowTitle(title);
        }
        is = VideoState::stream_open(this);
        cout << "stream opened" << endl;
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
        if (is->show_mode != SHOW_MODE_NONE && (!is->paused || is->force_refresh)) {
            is->video_refresh(&remaining_time);
        }
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
    QMainWindow::paintEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    //cout << QTime::currentTime().toString("hh:mm:ss.zzz").toStdString() << endl;
    QMainWindow::resizeEvent(event);
}

void MainWindow::moveEvent(QMoveEvent *event)
{
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
    //filterDialog->panel->saveSettings(settings);
    parameterDialog->panel->saveSettings(settings);

    filterDialog->closeEvent(event);

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

void MainWindow::openFile()
{
    QString default_path = QDir::homePath();

    QString path = QFileDialog::getOpenFileName(this, "", default_path, "");
    if (path.length() > 0) {
        co->input_filename = av_strdup(path.toLatin1().data());
        mainPanel->controlPanel->play();
    }
}

void MainWindow::menuAction(QAction *action)
{
    cout << "action text: " << action->text().toStdString() << endl;

    if (action->text() == "&Open" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_O))
        openFile();
    else if (action->text() == "E&xit" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_X))
        close();
    else if (action->text() == "&Play/Pause" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_P))
        mainPanel->controlPanel->play();
    else if (action->text() == "&Rewind" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_R))
        mainPanel->controlPanel->rewind();
    else if (action->text() == "Fas&t Forward" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_T))
        mainPanel->controlPanel->fastforward();
    else if (action->text() == "Pre&vious" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_V))
        mainPanel->controlPanel->previous();
    else if (action->text() == "&Next" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_N))
        mainPanel->controlPanel->next();
    else if (action->text() == "&Quit" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_Q))
        mainPanel->controlPanel->quit();
    else if (action->text() == "&Mute" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_M))
        mainPanel->controlPanel->mute();
    else if (action->text() == "&Filters" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_F))
        if (filterDialog->isVisible()) filterDialog->hide(); else filterDialog->show();
    else if (action->text() == "&Engage" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_E))
        filterDialog->panel->engageFilter->setChecked(!filterDialog->panel->engageFilter->isChecked());
    else if (action->text() == "&Set Parameters" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_S))
        if (parameterDialog->isVisible()) parameterDialog->hide(); else parameterDialog->show();
    else if (action->text() == "Messa&ges" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_G))
        if (messageBox->isVisible()) messageBox->hide(); else messageBox->show();
    else if (action->text() == "&Options")
        optionDialog->show();
}

void MainWindow::showHelp(const QString &str)
{
    cout << str.toStdString() << endl;
}

void MainWindow::ping(const vector<bbox_t>* arg)
{
    /*
    cout << "ping: " << arg->size() << endl;
    for (const bbox_t detection : *arg) {
        cout << " "  << detection.track_id;
    }
    cout << endl;
    */
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
