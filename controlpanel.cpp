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
    QPushButton *singlestep = new QPushButton("Single");
    QPushButton *mute = new QPushButton("Mute");
    QPushButton *volup = new QPushButton("^");
    QPushButton *voldn = new QPushButton("v");
    QPushButton *quit = new QPushButton("Quit");
    QPushButton *infer = new QPushButton("Infer");
    QPushButton *test = new QPushButton("Test");
    engageFilter = new QCheckBox("Engage Filter");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(play,         1,  0, 1, 1);
    layout->addWidget(rewind,       1,  1, 1, 1);
    layout->addWidget(fastforward,  1,  2, 1, 1);
    layout->addWidget(pause,        1,  3, 1, 1);
    layout->addWidget(singlestep,   1,  4, 1, 1);
    layout->addWidget(mute,         1,  5, 1, 1);
    layout->addWidget(volup,        1,  6, 1, 1);
    layout->addWidget(voldn,        1,  7, 1, 1);
    layout->addWidget(quit,         1,  8, 1, 1);
    layout->addWidget(test,         1,  9, 1, 1);
    layout->addWidget(infer,        1, 10, 1, 1);
    layout->addWidget(engageFilter, 1, 11, 1, 1);
    setLayout(layout);

    connect(play, SIGNAL(clicked()), mainWindow, SLOT(runLoop()));
    connect(rewind, SIGNAL(clicked()), this, SLOT(rewind()));
    connect(fastforward, SIGNAL(clicked()), this, SLOT(fastforward()));
    connect(pause, SIGNAL(clicked()), this, SLOT(pause()));
    connect(singlestep, SIGNAL(clicked()), this, SLOT(singlestep()));
    connect(mute, SIGNAL(clicked()), this, SLOT(mute()));
    connect(volup, SIGNAL(clicked()), this, SLOT(volup()));
    connect(voldn, SIGNAL(clicked()), this, SLOT(voldn()));
    connect(quit, SIGNAL(clicked()), this, SLOT(quit()));
    //connect(infer, SIGNAL(clicked()), mainWindow, SLOT(infer()));
    connect(test, SIGNAL(clicked()), this, SLOT(test()));

}

void ControlPanel::resizeEvent(QResizeEvent *event) {
    cout << "ControlPanel width: " << event->size().width() << " height: " << event->size().height() << endl;
}

void ControlPanel::test()
{
    cout << "ControlPanel::test" << endl;
    //MW->runLoop();

    //MW->co->video_codec_name = "h264_qsv";
    //MW->co->opt_add_vfilter(NULL, NULL, "fps=1");
    MW->test();
}

void ControlPanel::singlestep()
{
    if (MW->is)
        MW->is->step_to_next_frame();
}

void ControlPanel::volup()
{
    if (MW->is)
        MW->is->update_volume(1, SDL_VOLUME_STEP);
}

void ControlPanel::voldn()
{
    if (MW->is)
        MW->is->update_volume(-1, SDL_VOLUME_STEP);
}

void ControlPanel::rewind()
{
    if (MW->is)
        MW->is->rewind();
}

void ControlPanel::fastforward()
{
    if (MW->is)
        MW->is->fastforward();
}

void ControlPanel::pause()
{
    if (MW->is)
        MW->is->toggle_pause();
}

void ControlPanel::mute()
{
    if (MW->is)
        MW->is->toggle_mute();
}

void ControlPanel::quit()
{
    //SDL_Event event;
    //event.type = FF_QUIT_EVENT;
    //event.user.data1 = this;
    //SDL_PushEvent(&event);
    MW->e->looping = false;
    //MW->timer->stop();
}

