#include <string>
#include <fstream>
#include <iomanip>

#include "controlpanel.h"
#include "mainwindow.h"

ControlPanel::ControlPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    //setStyleSheet("QPushButton { width: 32; height: 32 }");

    icnPlay = QIcon(":play.png");
    icnPause = QIcon(":pause.png");
    btnPlay = new QPushButton(icnPlay, "");
    //btnPlay->setMinimumSize(QSize(32, 32));

    icnStop = QIcon(":stop.png");
    btnStop = new QPushButton(icnStop, "");

    icnRewind = QIcon(":rewind.png");
    btnRewind = new QPushButton(icnRewind, "");

    icnFastForward = QIcon(":fast-forward.png");
    btnFastForward = new QPushButton(icnFastForward, "");

    //QPushButton *test = new QPushButton("Test");

    engageFilter = new QCheckBox("Engage Filter");
    volumeSlider = new QSlider(Qt::Horizontal, mainWindow);
    volumeSlider->setRange(0, 128);
    volumeSlider->setValue(100);

    icnAudioOn = QIcon(":audio.png");
    icnAudioOff = QIcon(":mute.png");
    btnMute = new QPushButton(icnAudioOn, "");

    QWidget *volumePanel = new QWidget;
    QHBoxLayout *volumeLayout = new QHBoxLayout;
    volumeLayout->setContentsMargins(11, 0, 11, 0);
    volumeLayout->addWidget(btnMute);
    volumeLayout->addWidget(volumeSlider);
    volumePanel->setLayout(volumeLayout);
    volumePanel->setMaximumHeight(40);

    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(11, 0, 11, 0);
    layout->addWidget(btnPlay,         1,  0, 1, 1);
    layout->addWidget(btnStop,         1,  1, 1, 1);
    layout->addWidget(btnRewind,       1,  2, 1, 1);
    layout->addWidget(btnFastForward,  1,  3, 1, 1);
    layout->addWidget(engageFilter,    1,  4, 1, 1);
    layout->addWidget(new QLabel,      1,  5, 1, 1);
    layout->addWidget(volumePanel,     1,  6, 1, 1);
    //layout->addWidget(test,            1,  7, 1, 1);
    layout->setColumnStretch(5, 10);
    setLayout(layout);

    setMaximumHeight(32);

    connect(btnPlay, SIGNAL(clicked()), this, SLOT(play()));
    connect(btnStop, SIGNAL(clicked()), this, SLOT(quit()));
    connect(btnRewind, SIGNAL(clicked()), this, SLOT(rewind()));
    connect(btnFastForward, SIGNAL(clicked()), this, SLOT(fastforward()));
    connect(btnMute, SIGNAL(clicked()), this, SLOT(mute()));
    connect(volumeSlider, SIGNAL(sliderMoved(int)), this, SLOT(sliderMoved(int)));
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));
    //connect(test, SIGNAL(clicked()), this, SLOT(test()));

}

void ControlPanel::sliderMoved(int arg)
{
    cout << "slider moved: " << arg << endl;
    if (MW->is)
        MW->is->audio_volume = arg;
}

void ControlPanel::resizeEvent(QResizeEvent *event) {
    cout << "ControlPanel width: " << event->size().width() << " height: " << event->size().height() << endl;
}

void ControlPanel::play()
{
    QString selected_filename = "";
    FilePanel *filePanel = (FilePanel*)MW->tabWidget->currentWidget();
    QModelIndex index = filePanel->tree->currentIndex();
    if (index.isValid()) {
        selected_filename = filePanel->model->filePath(index);
    }

    if (!MW->co->input_filename) {
        if (selected_filename.length() > 0) {
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
        }
        else {
            QMessageBox::critical(mainWindow, "PlayQt", "No current file, please select a file from the playlist");
            return;
        }
    }

    if (selected_filename.length() > 0) {
        QString title = "PlayQt";
        QFileInfo fi(selected_filename);
        title = title + " - " + fi.fileName();
        MW->setWindowTitle(title);
    }

    if (stopped) {
        if (selected_filename.length() > 0 && selected_filename != MW->co->input_filename) {
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
        }
        stopped = false;
        paused = false;
        btnPlay->setIcon(icnPause);
        MW->runLoop();
    }
    else if (paused) {
        if (selected_filename.length() > 0 && selected_filename != MW->co->input_filename) {
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
            MW->runLoop();
        }
        else {
            MW->is->toggle_pause();
        }
        paused = false;
        btnPlay->setIcon(icnPause);
    }
    else {
        if (selected_filename.length() > 0 && selected_filename != MW->co->input_filename) {
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
            MW->runLoop();
        }
        else {
            MW->is->toggle_pause();
            paused = true;
            btnPlay->setIcon(icnPlay);
        }
    }
}

void ControlPanel::test()
{
    cout << "ControlPanel::test" << endl;
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

    if (MW->is)
        cout << MW->is->audio_volume << endl;
}

void ControlPanel::voldn()
{
    if (MW->is)
        MW->is->update_volume(-1, SDL_VOLUME_STEP);

    if (MW->is)
        cout << MW->is->audio_volume << endl;
}

void ControlPanel::rewind()
{
    //SDL_Event event;
    //SDL_memset(&event, 0, sizeof(event));
    //event.type = MW->sdlCustomEventType;
    //event.user.code = REWIND;
    //SDL_PushEvent(&event);

    if (MW->is)
        MW->is->rewind();

}

void ControlPanel::fastforward()
{
    //SDL_Event event;
    //SDL_memset(&event, 0, sizeof(event));
    //event.type = MW->sdlCustomEventType;
    //event.user.code = FASTFORWARD;
    //SDL_PushEvent(&event);

    if (MW->is)
        MW->is->fastforward();

}

void ControlPanel::pause()
{
    //SDL_Event event;
    //SDL_memset(&event, 0, sizeof(event));
    //event.type = MW->sdlCustomEventType;
    //event.user.code = PAUSE;
    //SDL_PushEvent(&event);
    if (MW->is)
        MW->is->toggle_pause();
}

void ControlPanel::mute()
{
    muted = !muted;

    if (muted) {
        if (MW->is)
            MW->is->audio_volume = 0;
        volumeSlider->setEnabled(false);
        btnMute->setIcon(icnAudioOff);
    }
    else {
        if (MW->is)
            MW->is->audio_volume = volumeSlider->value();
        volumeSlider->setEnabled(true);
        btnMute->setIcon(icnAudioOn);
    }

}

void ControlPanel::quit()
{
    if (MW->is)
        MW->is->abort_request = 1;

    stopped = true;
    paused = false;
    btnPlay->setIcon(icnPlay);
    MW->co->input_filename = nullptr;

    SDL_Event event;
    event.type = FF_QUIT_EVENT;
    event.user.data1 = this;
    SDL_PushEvent(&event);

    //MW->e->looping = false;
    //MW->timer->stop();
}

