#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    av_log_set_level(AV_LOG_PANIC);

    QScreen *screen = QApplication::primaryScreen();
    QRect screenSize = screen->geometry();
    cout << "width: " << screenSize.width() << " height: " << screenSize.height() << endl;
    int aw = 1300;
    int ah = 800;
    int cx = screenSize.center().x();
    int cy = screenSize.center().y();
    int ax = cx - aw/2;
    int ay = cy - ah/2;

    setWindowTitle("Play Qt");
    settings = new QSettings("PlayQt", "Program Settings");

    mainPanel = new MainPanel(this);
    setCentralWidget(mainPanel);

    names_file = settings->value("MainWindow/names_file", "").toString();
    cfg_file = settings->value("MainWindow/cfg_file", "").toString();
    weights_file = settings->value("MainWindow/weights_file", "").toString();
    initializeModelOnStartup = settings->value("MainWindow/initializeModelOnStartup", false).toBool();

    if (initializeModelOnStartup) {
        model = new Model(this);
        model->initialize(cfg_file, weights_file, names_file, 0);
    }

    modelConfigureDialog = new ModelConfigureDialog(this);


    filterDialog = new FilterDialog(this);
    optionDialog = new OptionDialog(this);

    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction("&Next");
    fileMenu->addAction("&Previous");
    fileMenu->addAction("&Save");
    fileMenu->addAction("&Quit");

    QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
    toolsMenu->addAction("&Model Options");
    toolsMenu->addAction("&Filters");
    toolsMenu->addAction("&Options");
    connect(fileMenu, SIGNAL(triggered(QAction*)), this, SLOT(fileMenuAction(QAction*)));
    connect(toolsMenu, SIGNAL(triggered(QAction*)), this, SLOT(toolsMenuAction(QAction*)));

    move(ax, ay);


    //test();
}

MainWindow::~MainWindow()
{

}

void MainWindow::printSizes()
{
    QSize windowSize = size();
    cout << "window width: " << windowSize.width() << " height: " << windowSize.height() << endl;

    QSize displaySize = mainPanel->displayContainer->display->size();
    cout << "display width: " << displaySize.width() << " height: " << displaySize.height() << endl;

    QSize controlSize = mainPanel->controlPanel->size();
    cout << "control width: " << controlSize.width() << " height: " << controlSize.height() << endl;
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
    cout << "test 3" << endl;
    display.window = SDL_CreateWindowFrom((void*)mainPanel->displayContainer->display->winId());
    display.renderer = SDL_CreateRenderer(display.window, -1, 0);
    SDL_GetRendererInfo(display.renderer, &display.renderer_info);

    if (!display.window || !display.renderer || !display.renderer_info.num_texture_formats) {
        av_log(NULL, AV_LOG_FATAL, "Failed to create window or renderer: %s", SDL_GetError());
    }
    cout << "test 4" << endl;
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QSize old_size = event->oldSize();
    QSize new_size = event->size();

    cout << "RESIZE old width: " << old_size.width() << " height: " << old_size.height() << endl;
    cout << "RESIZE new width: " << new_size.width() << " height: " << new_size.height() << endl;

    printSizes();

    if (mainPanel->displayContainer->windowHeightDifferential == -1) {
        mainPanel->displayContainer->windowHeightDifferential
                = event->size().height() - mainPanel->displayContainer->displayInitialHeight;
    }
    /*
    else {
        mainPanel->displayContainer->setDisplayHeight(event->size().height());
    }
    */


    cout << "resizeEvent done" << endl;
}

void MainWindow::moveEvent(QMoveEvent *event)
{
    QPoint old_pos = event->oldPos();
    QPoint new_pos = event->pos();

    cout << "MOVE old pos x: " << old_pos.x() << " y: " << old_pos.y() << endl;
    cout << "MOVE new pos x: " << new_pos.x() << " y: " << new_pos.y() << endl;
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
    else if (action->text() == "&Filters")
        filterDialog->show();
    else if (action->text() == "&Options")
        optionDialog->show();
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
    cout << "x: " <<geometry().x() << " y: " << geometry().y() << " w: " << geometry().width() << " h: " << geometry().height() << endl;
    printSizes();
    //is->filter->test();
}

