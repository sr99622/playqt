#include <string>
#include <fstream>
#include <iomanip>

#include "controlpanel.h"
#include "mainwindow.h"

ControlPanel::ControlPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;
    btnPlay = new QPushButton();
    btnStop = new QPushButton();
    btnRewind = new QPushButton();
    btnFastForward = new QPushButton();
    btnNext = new QPushButton();
    btnPrevious = new QPushButton();
    btnMute = new QPushButton();
    styleButtons();

    QLabel *spacer = new QLabel("     ");
    engageFilter = new QCheckBox("Engage Filter");

    volumeSlider = new QSlider(Qt::Horizontal, mainWindow);
    volumeSlider->setRange(0, 128);
    volumeSlider->setValue(100);


    QWidget *volumePanel = new QWidget;
    QHBoxLayout *volumeLayout = new QHBoxLayout;
    volumeLayout->setContentsMargins(11, 0, 11, 0);
    volumeLayout->addWidget(btnMute);
    volumeLayout->addWidget(volumeSlider);
    volumePanel->setLayout(volumeLayout);

    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(11, 0, 11, 11);
    layout->addWidget(btnPlay,         1,  0, 1, 1);
    layout->addWidget(btnStop,         1,  1, 1, 1);
    layout->addWidget(btnRewind,       1,  2, 1, 1);
    layout->addWidget(btnFastForward,  1,  3, 1, 1);
    layout->addWidget(btnPrevious,     1,  4, 1, 1);
    layout->addWidget(btnNext,         1,  5, 1, 1);
    layout->addWidget(spacer,          1,  6, 1, 1);
    layout->addWidget(engageFilter,    1,  7, 1, 1);
    layout->addWidget(new QLabel,      1,  8, 1, 1);
    layout->addWidget(volumePanel,     1,  9, 1, 1);
    layout->setColumnStretch(8, 10);
    setLayout(layout);

    connect(btnPlay, SIGNAL(clicked()), this, SLOT(play()));
    connect(btnStop, SIGNAL(clicked()), this, SLOT(quit()));
    connect(btnRewind, SIGNAL(clicked()), this, SLOT(rewind()));
    connect(btnFastForward, SIGNAL(clicked()), this, SLOT(fastforward()));
    connect(btnPrevious, SIGNAL(clicked()), this, SLOT(previous()));
    connect(btnNext, SIGNAL(clicked()), this, SLOT(next()));
    connect(btnMute, SIGNAL(clicked()), this, SLOT(mute()));
    connect(engageFilter, SIGNAL(stateChanged(int)), this, SLOT(engage(int)));
    connect(volumeSlider, SIGNAL(sliderMoved(int)), this, SLOT(sliderMoved(int)));
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));

}

void ControlPanel::styleButtons()
{

    if (!paused)
        btnPlay->setStyleSheet(getButtonStyle("pause"));
    else
        btnPlay->setStyleSheet(getButtonStyle("play"));
    btnStop->setStyleSheet(getButtonStyle("stop"));
    btnRewind->setStyleSheet(getButtonStyle("rewind"));
    btnFastForward->setStyleSheet(getButtonStyle("fast-forward"));
    btnNext->setStyleSheet(getButtonStyle("next"));
    btnPrevious->setStyleSheet(getButtonStyle("previous"));
    if (muted)
        btnMute->setStyleSheet(getButtonStyle("mute"));
    else
        btnMute->setStyleSheet(getButtonStyle("audio"));
}

QString ControlPanel::getButtonStyle(const QString& name) const
{
    if (MW->config()->useSystemGui->isChecked())
        return QString("QPushButton {image:url(:%1_lo.png);}").arg(name);
    else
        return QString("QPushButton {image:url(:%1.png);} QPushButton:hover {image:url(:%1_hi.png);} QPushButton:pressed {image:url(:%1.png);}").arg(name);
}

void ControlPanel::sliderMoved(int arg)
{
    cout << "slider moved: " << arg << endl;
    if (MW->is)
        MW->is->audio_volume = arg;
}

void ControlPanel::resizeEvent(QResizeEvent *event) {
    //cout << "ControlPanel width: " << event->size().width() << " height: " << event->size().height() << endl;
    QWidget::resizeEvent(event);
}

void ControlPanel::play()
{
    QString selected_filename = "";

    if (MW->tabWidget->tabText(MW->tabWidget->currentIndex()) == "Cameras") {
        CameraPanel *cameraPanel = (CameraPanel*)MW->tabWidget->currentWidget();
        Camera *camera = cameraPanel->cameraList->getCurrentCamera();
        if (camera) {
            QString rtsp = camera->onvif_data->stream_uri;
            QString username = camera->onvif_data->username;
            QString password = camera->onvif_data->password;
            selected_filename = rtsp.mid(0, 7) + username + ":" + password + "@" + rtsp.mid(7);
        }
    }
    else {
        FilePanel *filePanel = (FilePanel*)MW->tabWidget->currentWidget();
        QModelIndex index = filePanel->tree->currentIndex();
        if (index.isValid()) {
            QFileInfo info = filePanel->model->fileInfo(index);
            if (!info.isDir()) {
                selected_filename = filePanel->model->filePath(index);
            }
            else {
                return;
            }
        }
    }

    if (!MW->co->input_filename) {
        if (selected_filename.length() > 0) {
            if (!checkCodec(selected_filename))
                return;
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
        }
        else {
            QMessageBox::critical(mainWindow, "PlayQt", "No current file, please select a file from the playlist");
            return;
        }
    }

    if (stopped) {
        if (selected_filename.length() > 0 && selected_filename != MW->co->input_filename) {
            if (!checkCodec(selected_filename))
                return;
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
        }
        stopped = false;
        paused = false;
        btnPlay->setStyleSheet(getButtonStyle("pause"));
        MW->runLoop();
    }
    else if (paused) {
        if (selected_filename.length() > 0 && selected_filename != MW->co->input_filename) {
            if (!checkCodec(selected_filename))
                return;
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
            MW->runLoop();
        }
        else {
            MW->is->toggle_pause();
        }
        paused = false;
        btnPlay->setStyleSheet(getButtonStyle("pause"));
    }
    else {
        if (selected_filename.length() > 0 && selected_filename != MW->co->input_filename) {
            if (!checkCodec(selected_filename))
                return;
            MW->co->input_filename = av_strdup(selected_filename.toLatin1().data());
            MW->runLoop();
        }
        else {
            MW->is->toggle_pause();
            paused = true;
            btnPlay->setStyleSheet(getButtonStyle("play"));
        }
    }
}

bool ControlPanel::checkCodec(const QString& filename)
{
    if (MW->co->video_codec_name && filename != MW->co->input_filename) {

        cout << "HAS VIDEO CODEC" << endl;

        AVFormatContext *fmt_ctx = nullptr;
        AVStream *video;
        int video_stream;

        try {
            av.ck(avformat_open_input(&fmt_ctx, filename.toLatin1().data(), NULL, NULL), AOI);
            av.ck(avformat_find_stream_info(fmt_ctx, NULL), AFSI);
            av.ck(video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0), AFBS);
            video = fmt_ctx->streams[video_stream];
            const AVCodecDescriptor *cd = avcodec_descriptor_get(video->codecpar->codec_id);
            if (cd) {
                QString forced_codec_name = MW->co->video_codec_name;
                if (!forced_codec_name.contains(cd->name)) {
                    QString str;
                    QFileInfo fi(filename);
                    QTextStream(&str) << "User specified codec '" << forced_codec_name << "' may not support the codec '" << cd->name
                                      << "'\n found in the file '" << fi.fileName() << "'   Do you want to proceed anyway ?";
                    QMessageBox::StandardButton result = QMessageBox::question(MW, "PlayQt", str);
                    if (result == QMessageBox::No) {
                        if (fmt_ctx)
                            avformat_close_input(&fmt_ctx);
                        return false;
                    }
                }
            }
        }
        catch (AVException *e) {
            emit msg(QString("Unable to open format context %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
        }

        if (fmt_ctx)
            avformat_close_input(&fmt_ctx);
    }

    return true;
}

void ControlPanel::engage(int state)
{
    MW->filter()->engageFilter->setChecked(state);
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
    if (MW->is)
        MW->is->rewind();
}

void ControlPanel::fastforward()
{
    if (MW->is)
        MW->is->fastforward();
}

void ControlPanel::previous()
{
    if (MW->tabWidget->tabText(MW->tabWidget->currentIndex()) == "Cameras") {
        CameraPanel *cameraPanel = (CameraPanel*)MW->tabWidget->currentWidget();
        QModelIndex previous = cameraPanel->cameraList->previousIndex();
        if (previous.isValid()) {
            cout << "Valid previous index" << endl;
            cameraPanel->cameraList->setCurrentIndex(previous);
            MW->mainPanel->controlPanel->play();
        }
    }
    else {
        FilePanel *filePanel = (FilePanel*)MW->tabWidget->currentWidget();
        QModelIndex previous = filePanel->tree->indexAbove(filePanel->tree->currentIndex());
        if (previous.isValid()) {
            filePanel->tree->setCurrentIndex(previous);
            MW->mainPanel->controlPanel->play();
        }
    }
}

void ControlPanel::next()
{
    if (MW->tabWidget->tabText(MW->tabWidget->currentIndex()) == "Cameras") {
        CameraPanel *cameraPanel = (CameraPanel*)MW->tabWidget->currentWidget();
        QModelIndex next = cameraPanel->cameraList->nextIndex();
        if (next.isValid()) {
            cameraPanel->cameraList->setCurrentIndex(next);
            MW->mainPanel->controlPanel->play();
        }
    }
    else {
        FilePanel *filePanel = (FilePanel*)MW->tabWidget->currentWidget();
        QModelIndex next = filePanel->tree->indexBelow(filePanel->tree->currentIndex());
        if (next.isValid()) {
            filePanel->tree->setCurrentIndex(next);
            MW->mainPanel->controlPanel->play();
        }
    }
}

void ControlPanel::pause()
{
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
        btnMute->setStyleSheet(getButtonStyle("mute"));
    }
    else {
        if (MW->is)
            MW->is->audio_volume = volumeSlider->value();
        volumeSlider->setEnabled(true);
        btnMute->setStyleSheet(getButtonStyle("audio"));
    }

    emit muting(muted);
}

void ControlPanel::restoreEngaged()
{
    MW->filter()->engageFilter->setChecked(lastEngaged);
}

void ControlPanel::quit()
{
    if (MW->is)
        MW->is->abort_request = 1;

    lastEngaged = MW->filter()->engageFilter->isChecked();
    MW->filter()->engageFilter->setChecked(false);

    stopped = true;
    paused = false;
    btnPlay->setStyleSheet(getButtonStyle("play"));
    MW->co->input_filename = nullptr;

    SDL_Event event;
    event.type = FF_QUIT_EVENT;
    event.user.data1 = this;
    SDL_PushEvent(&event);

    emit quitting();
}

