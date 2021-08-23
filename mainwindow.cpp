#include "mainwindow.h"
#include "Ffplay/cmdutils.h"

using namespace cv::cuda;

MainWindow::MainWindow(CommandOptions *co, QWidget *parent) : QMainWindow(parent)
{
    this->co = co;
    co->mainWindow = this;
    filename = QString(co->input_filename);
    av_log_set_level(AV_LOG_PANIC);

    settings = new QSettings("PlayQt", "Program Settings");
    configDialog = new ConfigDialog(this);

    autoSaveTimer = new QTimer(this);
    connect(autoSaveTimer, SIGNAL(timeout()), this, SLOT(autoSave()));
    autoSaveTimer->start(10000);

    screen = QApplication::primaryScreen();
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

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(poll()));
    timer->start(1000 / 30);

    tabWidget = new QTabWidget(this);
    tabWidget->setMinimumWidth(100);

    videoPanel = new FilePanel(this, "Videos", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation));
    picturePanel = new FilePanel(this, "Pictures", QStandardPaths::writableLocation(QStandardPaths::PicturesLocation));
    audioPanel = new FilePanel(this, "Audio", QStandardPaths::writableLocation(QStandardPaths::MusicLocation));
    cameraPanel = new CameraPanel(this);
    tabWidget->addTab(videoPanel, tr("Videos"));
    tabWidget->addTab(picturePanel, tr("Pictures"));
    tabWidget->addTab(audioPanel, tr("Audio"));
    tabWidget->addTab(cameraPanel, tr("Cameras"));

    mainPanel = new MainPanel(this);

    splitter = new QSplitter(Qt::Orientation::Horizontal, this);
    splitter->addWidget(mainPanel);
    splitter->addWidget(tabWidget);
    if (settings->contains(splitterKey))
        splitter->restoreState(settings->value(splitterKey).toByteArray());
    connect(splitter, SIGNAL(splitterMoved(int, int)), this, SLOT(splitterMoved(int, int)));

    setCentralWidget(splitter);

    messageDialog = new MessageDialog(this);
    filterDialog = new FilterDialog(this);
    optionDialog = new OptionDialog(this);
    parameterDialog = new ParameterDialog(this);
    filterChain = new FilterChain(this);
    countDialog = new CountDialog(this);

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
    QAction *actCount = new QAction(tr("&Count"));
    actCount->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_C));
    toolsMenu->addAction(actCount);
    QAction *actConfig = new QAction(tr("Conf&ig"));
    toolsMenu->addAction(actConfig);

    QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(tr("&Options"));

    connect(fileMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(mediaMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(toolsMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(helpMenu, SIGNAL(triggered(QAction*)), this, SLOT(menuAction(QAction*)));
    connect(co, SIGNAL(showHelp(const QString&)), optionDialog->panel, SLOT(showConfig(const QString&)));

    applyStyle(config()->getProfile());
    control()->restoreEngageSetting();

    sdlCustomEventType = SDL_RegisterEvents(1);
    av_init_packet(&flush_pkt);
    flush_pkt.data = (uint8_t*)&flush_pkt;
    e = new EventHandler(this);
    timer->start();

    int startup_volume = av_clip(co->startup_volume, 0, 128);
    if (startup_volume == 0)
        control()->mute();
    else
        control()->volumeSlider->setValue(startup_volume);

    if (co->input_filename) {
        launcher = new Launcher(this);
        connect(launcher, SIGNAL(done()), control(), SLOT(play()));
        QThreadPool::globalInstance()->tryStart(launcher);
    }

}

void MainWindow::menuAction(QAction *action)
{
    if (action->text() == "&Open" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_O))
        openFile();
    else if (action->text() == "E&xit" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_X))
        close();
    else if (action->text() == "&Play/Pause" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_P))
        control()->play();
    else if (action->text() == "&Rewind" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_R))
        control()->rewind();
    else if (action->text() == "Fas&t Forward" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_T))
        control()->fastforward();
    else if (action->text() == "Pre&vious" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_V))
        control()->previous();
    else if (action->text() == "&Next" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_N))
        control()->next();
    else if (action->text() == "&Quit" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_Q))
        control()->quit();
    else if (action->text() == "&Mute" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_M))
        control()->mute();
    else if (action->text() == "&Filters" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_F))
        if (filterDialog->isVisible()) filterDialog->hide(); else filterDialog->show();
    else if (action->text() == "&Engage" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_E))
        (filter()->toggleEngage());
    else if (action->text() == "&Set Parameters" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_S))
        if (parameterDialog->isVisible()) parameterDialog->hide(); else parameterDialog->show();
    else if (action->text() == "Messa&ges" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_G))
        if (messageDialog->isVisible()) messageDialog->hide(); else messageDialog->show();
    else if (action->text() == "&Count" || action->shortcut() == QKeySequence(Qt::CTRL | Qt::Key_C))
        if (countDialog->isVisible()) countDialog->hide(); else countDialog->show();
    else if (action->text() == "&Options")
        optionDialog->show();
    else if (action->text() == "Conf&ig")
        configDialog->show();
}

void MainWindow::autoSave()
{
    if (changed) {
        cout << "MainWindow::autoSave" << endl;
        settings->setValue(geometryKey, saveGeometry());
        settings->setValue(splitterKey, splitter->saveState());
        changed = false;
    }

    videoPanel->autoSave();
    picturePanel->autoSave();
    audioPanel->autoSave();
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
            connect(quitter, SIGNAL(done()), control(), SLOT(play()));
        }
        QThreadPool::globalInstance()->tryStart(quitter);
    }
    else {
        start();
    }
}

void MainWindow::start()
{
    if (co->input_filename) {
        QString str;
        QTextStream(&str) << "start: " << co->input_filename << " at " << QTime::currentTime().toString("hh:mm:ss");
        msg(str);

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
        e->event_loop();
    }
}

void MainWindow::poll()
{
    if (is) {
        if (is->paused) {
            is->video_display();
        }

        double remaining_time = REFRESH_RATE;
        if (is->show_mode != SHOW_MODE_NONE && (!is->paused || is->force_refresh)) {
            is->video_refresh(&remaining_time);
        }
    }
}

void MainWindow::initializeSDL()
{
    ffDisplay.window = SDL_CreateWindowFrom((void*)display()->winId());
    ffDisplay.renderer = SDL_CreateRenderer(ffDisplay.window, -1, 0);
    SDL_GetRendererInfo(ffDisplay.renderer, &ffDisplay.renderer_info);

    if (!ffDisplay.window || !ffDisplay.renderer || !ffDisplay.renderer_info.num_texture_formats) {
        QString str;
        QTextStream(&str) << "MainWindow::initializeSDL error - Failed to create window or renderer: " << SDL_GetError();
        msg(str);
    }
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QMainWindow::paintEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    if (isVisible())
        changed = true;
    QMainWindow::resizeEvent(event);
}

void MainWindow::moveEvent(QMoveEvent *event)
{
    if (isVisible())
        changed = true;
    QMainWindow::moveEvent(event);
}

void MainWindow::showEvent(QShowEvent *event)
{
    QMainWindow::showEvent(event);
}

void MainWindow::splitterMoved(int pos, int index)
{
    if (isVisible())
        changed = true;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    autoSave();
    control()->quit();
    QMainWindow::closeEvent(event);
}

void MainWindow::msg(const QString &str)
{
    messageDialog->message->append(str);
    messageDialog->message->ensureCursorVisible();
}

void MainWindow::openFile()
{
    QString default_path = QDir::homePath();

    QString path = QFileDialog::getOpenFileName(this, "", default_path, "");
    if (path.length() > 0) {
        co->input_filename = av_strdup(path.toLatin1().data());
        control()->play();
    }
}

void MainWindow::applyStyle(const ColorProfile& profile)
{
    if (config()->useSystemGui->isChecked()) {
        setStyleSheet("");
        control()->styleButtons();
        filter()->styleButtons();
        display()->setStyleSheet("");
        parameter()->applyStyle(profile);
        return;
    }

    QFile f(":darkstyle.qss");
    if (!f.exists()) {
        msg("Error: MainWindow::getThemes() Style sheet not found");
    }
    else {
        f.open(QFile::ReadOnly | QFile::Text);
        style = QString(f.readAll());

        style.replace("background_light",  profile.bl);
        style.replace("background_medium", profile.bm);
        style.replace("background_dark",   profile.bd);
        style.replace("foreground_light",  profile.fl);
        style.replace("foreground_medium", profile.fm);
        style.replace("foreground_dark",   profile.fd);
        style.replace("selection_light",   profile.sl);
        style.replace("selection_medium",  profile.sm);
        style.replace("selection_dark",    profile.sd);

        setStyleSheet(style);
        control()->styleButtons();
        filter()->styleButtons();
        display()->setStyleSheet(QString("QFrame {background-color: %1; padding: 0px;}").arg(profile.bm));
        parameter()->applyStyle(profile);
    }

}

void MainWindow::showHelp(const QString &str)
{
    cout << str.toStdString() << endl;
}

ConfigPanel *MainWindow::config()
{
    return (ConfigPanel*)configDialog->panel;
}

ControlPanel *MainWindow::control()
{
    return mainPanel->controlPanel;
}

FilterPanel *MainWindow::filter()
{
    return (FilterPanel*)filterDialog->panel;
}

ParameterPanel *MainWindow::parameter()
{
    return (ParameterPanel*)parameterDialog->panel;
}

CountPanel *MainWindow::count()
{
    return (CountPanel*)countDialog->panel;
}

QLabel *MainWindow::display()
{
    return mainPanel->displayContainer->display;
}

DisplayContainer *MainWindow::dc()
{
    return mainPanel->displayContainer;
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

    MW->control()->stopped = true;
    MW->control()->paused = false;

    emit done();
}
