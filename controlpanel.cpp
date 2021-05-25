#include <string>
#include <fstream>
#include <iomanip>

#include "controlpanel.h"
#include "mainwindow.h"

ControlPanel::ControlPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    QPushButton *play = new QPushButton("Play");
    QPushButton *rewind = new QPushButton("<<");
    QPushButton *fastforward = new QPushButton(">>");
    QPushButton *pause = new QPushButton("Pause");
    QPushButton *mute = new QPushButton("Mute");
    QPushButton *quit = new QPushButton("Quit");
    QPushButton *infer = new QPushButton("Infer");
    QPushButton *test = new QPushButton("Test");
    engageFilter = new QCheckBox("Engage Filter");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(play,         0, 0, 1, 1, Qt::AlignCenter);
    layout->addWidget(rewind,       0, 1, 1, 1, Qt::AlignCenter);
    layout->addWidget(fastforward,  0, 2, 1, 1, Qt::AlignCenter);
    layout->addWidget(pause,        0, 3, 1, 1, Qt::AlignCenter);
    layout->addWidget(mute,         0, 4, 1, 1, Qt::AlignCenter);
    layout->addWidget(quit,         0, 5, 1, 1, Qt::AlignCenter);
    layout->addWidget(test,         0, 6, 1, 1, Qt::AlignCenter);
    layout->addWidget(infer,        0, 7, 1, 1, Qt::AlignCenter);
    layout->addWidget(engageFilter, 0, 8, 1, 1, Qt::AlignCenter);
    setLayout(layout);

    connect(play, SIGNAL(clicked()), mainWindow, SLOT(runLoop()));
    connect(rewind, SIGNAL(clicked()), this, SLOT(rewind()));
    connect(fastforward, SIGNAL(clicked()), this, SLOT(fastforward()));
    connect(pause, SIGNAL(clicked()), this, SLOT(pause()));
    connect(mute, SIGNAL(clicked()), this, SLOT(mute()));
    connect(quit, SIGNAL(clicked()), this, SLOT(quit()));
    //connect(infer, SIGNAL(clicked()), mainWindow, SLOT(infer()));
    connect(test, SIGNAL(clicked()), this, SLOT(test()));

}

void ControlPanel::test()
{
    cout << "ControlPanel::test" << endl;
    //MW->runLoop();

    //MW->co->video_codec_name = "h264_qsv";
    //MW->co->opt_add_vfilter(NULL, NULL, "fps=1");
    MW->test();
}

void ControlPanel::rewind()
{
    SDL_Event event;
    event.type = SDL_KEYDOWN;
    event.key.keysym.sym = SDLK_LEFT;
    int result = SDL_PushEvent(&event);
    if (result < 0)
        cout << "SDL Push Event Failure" << endl;
}

void ControlPanel::fastforward()
{
    SDL_Event event;
    event.type = SDL_KEYDOWN;
    event.key.keysym.sym = SDLK_RIGHT;
    int result = SDL_PushEvent(&event);
    if (result < 0)
        cout << "SDL Push Event Failure" << endl;
}

void ControlPanel::pause()
{
    SDL_Event event;
    event.type = SDL_KEYDOWN;
    event.key.keysym.sym = SDLK_SPACE;
    int result = SDL_PushEvent(&event);
    if (result < 0)
        cout << "SDL Push Event Failure" << endl;
}

void ControlPanel::mute()
{
    SDL_Event event;
    event.type = SDL_KEYDOWN;
    event.key.keysym.sym = SDLK_m;
    int result = SDL_PushEvent(&event);
    if (result < 0)
        cout << "SDL Push Event Failure" << endl;
}

void ControlPanel::quit()
{
    SDL_Event event;
    event.type = SDL_KEYDOWN;
    event.key.keysym.sym = SDLK_ESCAPE;
    int result = SDL_PushEvent(&event);
    if (result < 0)
        cout << "SLD Push Event Failure" << endl;
}

